#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

from poses.ransac import EstimatorType, RANSACEstimator


class Matches:
    """
    A class to store matched keypoints and their indices between two sets of keypoints.
    """
    def __init__(self, kpts, kpts_other, idx, idx_other):
        self.kpts = kpts
        self.kpts_other = kpts_other
        self.idx = idx
        self.idx_other = idx_other


# Adapted from https://github.com/verlab/accelerated_features
def match(feats1, feats2, min_cossim=0.82):
    cossim = feats1 @ feats2.t()

    bestcossim, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)

    idx0 = torch.arange(match12.shape[0], device=match12.device)
    mask = match21[match12] == idx0

    if min_cossim > 0:
        mask *= bestcossim > min_cossim

    return idx0, match12, mask


class Matcher:
    @torch.no_grad()
    def __init__(
        self,
        fundmat_samples: int,
        max_error: float,
        matcher_type: str = "mnn",
        feature_type: str = "xfeat",
        width: int = None,
        height: int = None,
        min_cossim: float = -1,
        keyframe_check_matcher: str = "mnn",
        lightglue_filter_threshold: float = 0.1,
    ):
        """
        Initialize the Matcher.
        Args:
            fundmat_samples (int): Number of RANSAC etimations when estimating inliers with fundamental matrix estimation.
            max_error (float): Maximum error for RANSAC inlier threshold.
            matcher_type (str): "mnn" (mutual nearest neighbor on descriptors) or "lightglue".
            feature_type (str): "xfeat" or "superpoint", used to select the LightGlue weights and the default MNN threshold.
            width, height (int): Image size, used by LightGlue to normalize keypoints.
            min_cossim (float): Minimum cosine similarity for MNN. If negative, a default is chosen based on feature_type.
            keyframe_check_matcher (str): Matcher used in evaluate_match. "mnn" or "same" (uses matcher_type).
            lightglue_filter_threshold (float): LightGlue match confidence threshold.
        """
        self.max_error = max_error
        self.fundmat_estimator = RANSACEstimator(
            fundmat_samples, max_error, EstimatorType.FUNDAMENTAL_8PTS
        )
        self.matcher_type = matcher_type
        self.eval_matcher_type = matcher_type if keyframe_check_matcher == "same" else "mnn"
        if min_cossim < 0:
            min_cossim = 0.75 if feature_type == "superpoint" else 0.82
        self.min_cossim = min_cossim

        if matcher_type == "lightglue":
            if feature_type != "superpoint":
                raise ValueError("LightGlue matching is only supported with SuperPoint features")
            try:
                from lightglue import LightGlue
            except ImportError as e:
                raise ImportError(
                    "LightGlue requires the lightglue package: pip install git+https://github.com/cvg/LightGlue.git"
                ) from e
            print("Loading LightGlue matcher")
            self.lightglue = LightGlue(
                features="superpoint", filter_threshold=lightglue_filter_threshold
            ).eval().cuda()
            assert width is not None and height is not None, "LightGlue requires the image size"
            self.image_size = torch.tensor([[width, height]], dtype=torch.float, device="cuda")
        elif matcher_type != "mnn":
            raise ValueError(f"Unknown matcher type: {matcher_type}")

    @torch.no_grad()
    def _lightglue_match(
        self, desc_kpts: 'DescribedKeypoints', desc_kpts_other: 'DescribedKeypoints'
    ):
        """
        Match with LightGlue using only valid keypoints, returning indices into the full keypoint arrays.
        """
        valid_idx = torch.nonzero(desc_kpts.valid.cuda())[:, 0]
        valid_idx_other = torch.nonzero(desc_kpts_other.valid.cuda())[:, 0]
        if len(valid_idx) == 0 or len(valid_idx_other) == 0:
            empty = torch.zeros(0, dtype=torch.long, device="cuda")
            return empty, empty.clone()

        data = {
            "image0": {
                "keypoints": desc_kpts.kpts.cuda()[valid_idx][None].float(),
                "descriptors": desc_kpts.feats.cuda()[valid_idx][None].float(),
                "image_size": self.image_size,
            },
            "image1": {
                "keypoints": desc_kpts_other.kpts.cuda()[valid_idx_other][None].float(),
                "descriptors": desc_kpts_other.feats.cuda()[valid_idx_other][None].float(),
                "image_size": self.image_size,
            },
        }
        matches = self.lightglue(data)["matches"][0]
        return valid_idx[matches[:, 0]], valid_idx_other[matches[:, 1]]

    @torch.no_grad()
    def _match_indices(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        matcher_type: str,
    ):
        """
        Returns the indices of the matched keypoints in both sets.
        """
        if matcher_type == "lightglue":
            return self._lightglue_match(desc_kpts, desc_kpts_other)
        idx, idx_other, mask = match(
            desc_kpts.feats.cuda(), desc_kpts_other.feats.cuda(), self.min_cossim
        )
        return idx[mask], idx_other[mask]

    def evaluate_match(
        self, desc_kpts: 'DescribedKeypoints', desc_kpts_other: 'DescribedKeypoints'
    ):
        """
        Get the number of matches between two sets of described keypoints.
        """
        idx, _ = self._match_indices(desc_kpts, desc_kpts_other, self.eval_matcher_type)
        return len(idx)

    @torch.no_grad()
    def __call__(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        remove_outliers: bool = False,
        update_kpts_flag: str = "",
        kID: int = -1,
        kID_other: int = -1,
    ):
        """
        Matches keypoints between two sets of described keypoints, with optional outlier removal based on the fundamental RANSAC estimation.
        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            remove_outliers (bool): Whether to remove outliers using the fundamental matrix.
            update_kpts_flag (str): If "all", updates all matches; if "inliers", updates only inliers.
            kID (int): ID of the first set of keypoints, used for updating matches.
            kID_other (int): ID of the second set of keypoints, used for updating matches.
        Returns:
            Matches: A Matches object containing the matched keypoints and their indices.
        """
        idx, idx_other = self._match_indices(
            desc_kpts, desc_kpts_other, self.matcher_type
        )
        kpts = desc_kpts.kpts[idx]
        kpts_other = desc_kpts_other.kpts[idx_other]
        idx_all = idx
        idx_other_all = idx_other
        kpts_all = kpts
        kpts_other_all = kpts_other

        if remove_outliers:
            F, mask = self.fundmat_estimator(kpts, kpts_other)
            idx = idx[mask]
            idx_other = idx_other[mask]
            kpts = kpts[mask]
            kpts_other = kpts_other[mask]

        if update_kpts_flag == "all":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts_all, kpts_other_all, idx_all, idx_other_all)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other_all, kpts_all, idx_other_all, idx_all)
            )
        elif update_kpts_flag == "inliers":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts, kpts_other, idx, idx_other)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other, kpts, idx_other, idx)
            )

        return Matches(kpts, kpts_other, idx, idx_other)
