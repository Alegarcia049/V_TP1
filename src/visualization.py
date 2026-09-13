import cv2
import matplotlib.pyplot as plt

from src.features import FeatureSet


def draw_keypoints(
    image,
    features: FeatureSet,
    rich: bool = True,
):
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if rich
        else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    return cv2.drawKeypoints(
        image,
        features.keypoints,
        None,
        flags=flags,
    )


def plot_features(
    image,
    features: FeatureSet,
    title: str | None = None,
):
    rendered = draw_keypoints(image, features)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    if title is not None:
        ax.set_title(
            f"{title} — {len(features.keypoints)} keypoints"
        )

    return fig

def plot_feature_comparison(
    image,
    before: FeatureSet,
    after: FeatureSet,
    titles=("Detected features", "After ANMS"),
):
    before_image = draw_keypoints(image, before)
    after_image = draw_keypoints(image, after)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, rendered, features, title in zip(
        axes,
        (before_image, after_image),
        (before, after),
        titles,
    ):
        ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title} — {len(features.keypoints)} keypoints")
        ax.axis("off")

    return fig

def plot_sift_descriptor(
    features: FeatureSet,
    index: int,
):
    descriptor = features.descriptors[index]
    histogram = descriptor.reshape(16, 8)

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(
        histogram,
        aspect="auto",
    )

    ax.set_title(f"SIFT descriptor — keypoint {index}")
    ax.set_xlabel("Orientation bin")
    ax.set_ylabel("Spatial cell")

    fig.colorbar(image, ax=ax, label="Magnitude")

    return fig

def plot_keypoint_distribution(
    image,
    features: FeatureSet,
    title: str | None = None,
):
    xs = [kp.pt[0] for kp in features.keypoints]
    ys = [kp.pt[1] for kp in features.keypoints]

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

    return fig