import numpy as np

def anms(
    keypoints: list,
    max_keypoints: int,
    local_radius: float,
) -> list:
    keypoints = _local_maxima(keypoints, local_radius)

    radii = _suppression_radii(keypoints)

    indices = np.argsort(radii)[::-1]

    return [keypoints[i] for i in indices[:max_keypoints]]

def _local_maxima(
    keypoints: list,
    radius: float,
) -> list:

    alive = []
    radius_sq = radius ** 2

    for ki in keypoints:
        xi, yi = ki.pt

        for kj in keypoints:
            if kj is ki:
                continue

            xj, yj = kj.pt
            distance_sq = (xj - xi) ** 2 + (yj - yi) ** 2

            if distance_sq <= radius_sq and kj.response > ki.response:
                break
        else:
            alive.append(ki)

    return alive    

def _suppression_radii(
    keypoints: list,
) -> np.ndarray:
    radii = np.full(len(keypoints), np.inf)

    for i, ki in enumerate(keypoints):
        xi, yi = ki.pt

        for kj in keypoints:
            if kj.response < ki.response or kj is ki:
                continue

            xj, yj = kj.pt

            distance_sq = (xj - xi) ** 2 + (yj - yi) ** 2
            radii[i] = min(radii[i], distance_sq)

    return radii