import numpy as np


def normalize_points(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    mean_distance = np.mean(
        np.linalg.norm(centered, axis=1)
    )

    scale = np.sqrt(2) / mean_distance

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])

    normalized = centered * scale

    return normalized, T


def _build_dlt_matrix(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:

    n = len(src_points)
    A = np.zeros((2 * n, 9), dtype=np.float64)

    for i, ((x, y), (u, v)) in enumerate(
        zip(src_points, dst_points)
    ):
        A[2 * i] = [
            -x, -y, -1,
            0, 0, 0,
            x * u, y * u, u,
        ]

        A[2 * i + 1] = [
            0, 0, 0,
            -x, -y, -1,
            x * v, y * v, v,
        ]

    return A


def dlt_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:

    src_normalized, T_src = normalize_points(src_points)
    dst_normalized, T_dst = normalize_points(dst_points)

    A = _build_dlt_matrix(
        src_normalized,
        dst_normalized,
    )

    _, _, Vt = np.linalg.svd(A)

    H_normalized = Vt[-1].reshape(3, 3)

    H = (
        np.linalg.inv(T_dst)
        @ H_normalized
        @ T_src
    )

    return H


def project_points(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:

    homogeneous = np.column_stack([
        points,
        np.ones(len(points)),
    ])

    projected = (H @ homogeneous.T).T

    return projected[:, :2] / projected[:, 2, None]


def reprojection_errors(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:

    projected = project_points(src_points, H)

    return np.linalg.norm(
        projected - dst_points,
        axis=1,
    )