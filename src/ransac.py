from dataclasses import dataclass
import numpy as np

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
    ...
    best_mask = ...

repetir T veces:
    elegir 4 correspondencias
    estimar H con DLT
    calcular error de todas
    identificar inliers
    conservar el consenso más grande

usar todos los mejores inliers
        ↓
DLT nuevamente
        ↓
H final

recalcular errores con H final