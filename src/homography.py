import numpy as np


def normalize_points(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Traslada el centroide de los puntos al origen 
    y escala para que la distancia media de los puntos
    al origen sea de sqrt(2), es decir que todos entren
    en un círculo unitario centrado al origen.
    Es una mejora de estabilidad numérica para usar SVD.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    mean_distance = np.mean(
        np.linalg.norm(centered, axis=1)
    )

    scale = np.sqrt(2) / mean_distance

    # Entonces la matriz de normalización
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

    # De forma stackeada, todas las matrices A_i del sistema lineal homogéneo
    for i, ((x, y), (x_prima, y_prima)) in enumerate(
        zip(src_points, dst_points)
    ):
        # la primera columna de A_i
        A[2 * i] = [
            -x, -y, -1,
            0, 0, 0,
            x * x_prima, y * x_prima, x_prima,
        ]

        # la segunda columna de A_i
        A[2 * i + 1] = [
            0, 0, 0,
            -x, -y, -1,
            x * y_prima, y * y_prima, y_prima,
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

    # La solución por SVD para LST resulta el autovector de menor valor singular
    H_normalized = Vt[-1].reshape(3, 3)

    # Como los puntos fueron normalizados, hay que reconstruir la homografía original
    # considerando T_dst @ H = H_normalized @ T_src
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

    cartesian = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64,
    )

    np.divide(
        projected[:, :2],
        projected[:, 2, None],
        out=cartesian,
        where=projected[:, 2, None] != 0,
    )

    return cartesian


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