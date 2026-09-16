from dataclasses import dataclass
import numpy as np
from src.homography import dlt_homography, reprojection_errors


@dataclass
class RansacResult:
    H: np.ndarray
    inlier_mask: np.ndarray
    errors: np.ndarray


def ransac_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float,
    max_iterations: int,
    rng: np.random.Generator,
) -> RansacResult:

    best_mask = np.zeros(len(src_points), dtype=bool)
    best_count = 0

    for _ in range(max_iterations):
        sample_indices = rng.choice(
            len(src_points),
            size=4,
            replace=False,
        )

        H = dlt_homography(
            src_points[sample_indices],
            dst_points[sample_indices],
        )

        errors = reprojection_errors(
            src_points,
            dst_points,
            H,
        )

        inlier_mask = errors < threshold
        inlier_count = np.count_nonzero(inlier_mask)

        if inlier_count > best_count:
            best_count = inlier_count
            best_mask = inlier_mask

    H = dlt_homography(
        src_points[best_mask],
        dst_points[best_mask],
    )

    errors = reprojection_errors(
        src_points,
        dst_points,
        H,
    )

    inlier_mask = errors < threshold

    return RansacResult(
        H=H,
        inlier_mask=inlier_mask,
        errors=errors,
    )