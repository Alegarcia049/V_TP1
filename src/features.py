from dataclasses import dataclass

import cv2
import numpy as np

from src.anms import anms


@dataclass
class FeatureSet:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


class SIFT:

    def __init__(self, max_keypoints: int):
        self.max_keypoints = max_keypoints
        self.algorithm = cv2.SIFT_create()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def extract(self, image: np.ndarray) -> FeatureSet:
        gray = self.preprocess(image)

        keypoints = self.algorithm.detect(gray, None)
        keypoints = anms(keypoints, self.max_keypoints)
        keypoints, descriptors = self.algorithm.compute(gray, keypoints)

        return FeatureSet(keypoints, descriptors)