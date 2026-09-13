def normalize_points(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ...


def dlt_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    ...


def project_points(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    ...


def reprojection_errors(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    ...

def _build_dlt_matrix(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    ...