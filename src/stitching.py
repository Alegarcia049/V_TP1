from dataclasses import dataclass
import cv2
import numpy as np
from src.homography import project_points


@dataclass
class CanvasGeometry:
    bounds: tuple[int, int, int, int]
    size: tuple[int, int]
    translation: np.ndarray


def image_corners(
    image_shape: tuple[int, int],
) -> np.ndarray:

    height, width = image_shape

    return np.array([
        [0, 0],             # left lower
        [width, 0],         # right lower
        [width, height],    # right upper
        [0, height],        # left upper
    ], dtype=np.float64)


def compute_canvas(
    image_shapes: list[tuple[int, int]],
    homographies: list[np.ndarray],
) -> CanvasGeometry:

    transformed_corners = [
        project_points(
            image_corners(shape),
            H,
        )
        for shape, H in zip(image_shapes, homographies)
    ]

    corners = np.vstack(transformed_corners)

    x_min = int(np.floor(corners[:, 0].min()))
    y_min = int(np.floor(corners[:, 1].min()))
    x_max = int(np.ceil(corners[:, 0].max()))
    y_max = int(np.ceil(corners[:, 1].max()))

    translation = np.array([    # traslada el origen de la imagen hacia el origen del lienzo
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1],
    ], dtype=np.float64)

    return CanvasGeometry(
        bounds=(x_min, y_min, x_max, y_max),
        size=(x_max - x_min, y_max - y_min),
        translation=translation,
    )


def warp_images(
    images: list[np.ndarray],
    homographies: list[np.ndarray],
    canvas: CanvasGeometry,
) -> tuple[list[np.ndarray], list[np.ndarray]]:

    warped_images = []
    warped_masks = []

    for image, H in zip(images, homographies):
        H_canvas = canvas.translation @ H
        # proyectar sobre H_canvas resulta como proyectar sobre H y luego trasladar

        warped_image = cv2.warpPerspective(
            image,
            H_canvas,
            canvas.size,
            flags=cv2.INTER_LINEAR,
        )

        # necesitamos una máscara binaria que diga donde va a caer
        # la imagen proyectada sobre el lienzo
        mask = np.ones(
            image.shape[:2],
            dtype=np.uint8,
        )

        warped_mask = cv2.warpPerspective(
            mask,
            H_canvas,
            canvas.size,
            flags=cv2.INTER_NEAREST,
        )

        warped_images.append(warped_image)
        warped_masks.append(warped_mask)

    return warped_images, warped_masks


def compute_blending_weights(
    masks: list[np.ndarray],
) -> list[np.ndarray]:

    weights = []

    for mask in masks:
        # antes de tomar distancias padeamos las máscaras para poder
        # calcular los pesos sobre el borde de la máscara,
        # sin esto se puede complicar el caso de que la máscara llegue
        # hasta el borde del lienzo
        padded = np.pad(
            mask,
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        distance = cv2.distanceTransform(
            padded,
            cv2.DIST_L2,
            5,
        )

        # El padding no nos sirve para la distancia final
        # queremos quedarnos con lo que está dentro
        weights.append(distance[1:-1, 1:-1])

    return weights


def blend_images(
    warped_images: list[np.ndarray],
    weights: list[np.ndarray],
) -> np.ndarray:

    accumulator = np.zeros_like(
        warped_images[0],
        dtype=np.float32,
    )

    total_weight = np.zeros(
        warped_images[0].shape[:2],
        dtype=np.float32,
    )

    for image, weight in zip(warped_images, weights):
        accumulator += (
            image.astype(np.float32)
            * weight[..., None]
        )

        total_weight += weight

    result = np.zeros_like(accumulator)

    np.divide(
        accumulator,
        total_weight[..., None],
        out=result,
        where=total_weight[..., None] > 0,
    )

    return np.clip(result, 0, 255).astype(np.uint8)