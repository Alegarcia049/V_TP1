from dataclasses import dataclass
import numpy as np

@dataclass
class CanvasGeometry:
    bounds: tuple[float, float, float, float]
    size: tuple[int, int]
    translation: np.ndarray


def image_corners(
    image_shape: tuple[int, int],
) -> np.ndarray:
    ...


def compute_canvas(
    image_shapes: list[tuple[int, int]],
    homographies: list[np.ndarray],
) -> CanvasGeometry:
    ...


def warp_images(
    images: list[np.ndarray],
    homographies: list[np.ndarray],
    canvas: CanvasGeometry,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ...


def compute_blending_weights(
    masks: list[np.ndarray],
) -> list[np.ndarray]:
    ...


def blend_images(
    warped_images: list[np.ndarray],
    weights: list[np.ndarray],
) -> np.ndarray:
    ...