from dataclasses import dataclass

import cv2
import numpy as np

from src.anms import anms


@dataclass
class FeatureSet:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


class SIFT:
    def __init__(self):
        self.algorithm = cv2.SIFT_create()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect(self, image: np.ndarray) -> list[cv2.KeyPoint]:
        image = self.preprocess(image)
        return self.algorithm.detect(image, None)

    def describe(
        self,
        image: np.ndarray,
        keypoints: list[cv2.KeyPoint],
    ) -> FeatureSet:
        image = self.preprocess(image)

        keypoints, descriptors = self.algorithm.compute(image, keypoints)

        return FeatureSet(
            keypoints=keypoints,
            descriptors=descriptors,
        )