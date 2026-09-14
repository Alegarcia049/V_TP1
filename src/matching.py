import cv2
import numpy as np


def match_cross_check(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
) -> list[cv2.DMatch]:

    matcher = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=True,
    )

    return matcher.match(descriptors_src, descriptors_dst)


def match_lowe_ratio(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:

    matcher = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=False,
    )

    candidates = matcher.knnMatch(
        descriptors_src,
        descriptors_dst,
        k=2,
    )

    return [
        best
        for best, second in candidates
        if best.distance < ratio * second.distance
    ]


def match_lowe_cross_check(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:

    forward_matches = match_lowe_ratio(
        descriptors_src,
        descriptors_dst,
        ratio,
    )

    matcher = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=False,
    )

    reverse_matches = matcher.match(
        descriptors_dst,
        descriptors_src,
    )

    reverse_best = {
        match.queryIdx: match.trainIdx
        for match in reverse_matches
    }

    return [
        match
        for match in forward_matches
        if reverse_best[match.trainIdx] == match.queryIdx
    ]


def matched_points(
    keypoints_src: list[cv2.KeyPoint],
    keypoints_dst: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:

    points_src = np.array(
        [keypoints_src[match.queryIdx].pt for match in matches],
        dtype=np.float64,
    )

    points_dst = np.array(
        [keypoints_dst[match.trainIdx].pt for match in matches],
        dtype=np.float64,
    )

    return points_src, points_dst