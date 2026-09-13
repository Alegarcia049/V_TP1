def match_cross_check(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    norm_type: int,
) -> list:
    cv2.BFMatcher(
    normType=norm_type,
    crossCheck=True,
)
    ...


def match_lowe_ratio(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    norm_type: int,
    ratio: float,
) -> list:
    ...


def matched_points(
    keypoints_src: list,
    keypoints_dst: list,
    matches: list,
) -> tuple[np.ndarray, np.ndarray]:
    ...

def match_lowe_cross_check(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    norm_type: int,
    ratio: float,
) -> list:
    ...