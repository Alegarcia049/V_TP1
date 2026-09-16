import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.features import FeatureSet


def draw_keypoints(
    image,
    keypoints,
    rich: bool = True,
):
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if rich
        else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(255, 0, 0),
        flags=flags,
    )


def plot_features(
    image,
    features: FeatureSet,
    title: str | None = None,
):
    rendered = draw_keypoints(
        image,
        features.keypoints,
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    if title is not None:
        ax.set_title(
            f"{title} — {len(features.keypoints)} keypoints"
        )


def plot_keypoint_comparison(
    image,
    before,
    after,
    titles=("Detected keypoints", "After ANMS"),
):
    before_image = draw_keypoints(image, before)
    after_image = draw_keypoints(image, after)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, rendered, keypoints, title in zip(
        axes,
        (before_image, after_image),
        (before, after),
        titles,
    ):
        ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{title} — {len(keypoints)} keypoints"
        )
        ax.axis("off")


def plot_sift_descriptor(
    features: FeatureSet,
    index: int,
):
    descriptor = features.descriptors[index]

    grid_size = 4
    orientation_bins = 8

    histogram = descriptor.reshape(
        grid_size,
        grid_size,
        orientation_bins,
    )

    max_magnitude = histogram.max()

    fig, ax = plt.subplots(figsize=(7, 7))

    for row in range(grid_size):
        for col in range(grid_size):
            center_x = col + 0.5
            center_y = row + 0.5

            for bin_index, magnitude in enumerate(histogram[row, col]):
                angle = 2 * np.pi * bin_index / orientation_bins
                length = 0.45 * magnitude / max_magnitude

                dx = length * np.cos(angle)
                dy = length * np.sin(angle)

                ax.plot(
                    [center_x - dx, center_x + dx],
                    [center_y - dy, center_y + dy],
                    linewidth=1.5,
                )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_aspect("equal")

    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)

    ax.set_title(f"SIFT descriptor — keypoint {index}")
    ax.set_xlabel("Spatial cell x")
    ax.set_ylabel("Spatial cell y")


def plot_keypoint_distribution(
    image,
    keypoints,
    title: str | None = None,
):
    xs = [kp.pt[0] for kp in keypoints]
    ys = [kp.pt[1] for kp in keypoints]

    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(xs, ys, s=10)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

def draw_matches(
    image_src,
    image_dst,
    keypoints_src: list,
    keypoints_dst: list,
    matches: list,
    max_matches: int | None = None,
):
    if max_matches is not None:
        matches = sorted(matches, key=lambda match: match.distance)[:max_matches]

    return cv2.drawMatches(
        image_src,
        keypoints_src,
        image_dst,
        keypoints_dst,
        matches,
        None,
        matchColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

def plot_matches(
    image_src,
    image_dst,
    keypoints_src: list,
    keypoints_dst: list,
    matches: list,
    max_matches: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (16, 8),
):
    rendered = draw_matches(
        image_src,
        image_dst,
        keypoints_src,
        keypoints_dst,
        matches,
        max_matches=max_matches,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    if title is None:
        title = f"Matches: {len(matches)}"

    if max_matches is not None:
        title += f" (showing {min(max_matches, len(matches))} matches)"

    ax.set_title(title)
    plt.show()