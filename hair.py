import mediapipe as mp
import cv2
import numpy as np

mp_hs = mp.solutions.hair_segmentation


def max_intensity_criterion(pixel) -> bool:
    b, g, r = pixel[0], pixel[1], pixel[2]
    assert g == 0
    assert r == 0
    if b == 254:
        return True
    return False


def eps_criterion(pixel, sensitivity=240) -> bool:
    b, g, r = pixel[0], pixel[1], pixel[2]
    assert g == 0
    assert r == 0
    eps = 0
    eps += b
    eps += g
    eps += r
    if eps > sensitivity:
        return True
    return False


class Hair:
    def __init__(self, criterion):
        self.criterion = criterion

    def get_boolean_mask(self, image: np.ndarray) -> np.ndarray:
        with mp_hs.HairSegmentation() as HS:
            results = HS.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        hair_mask_bgr = results.hair_mask
        hair_mask_bgr = cv2.resize(hair_mask_bgr, (image.shape[1], image.shape[0]))

        boolean_mask = np.zeros((hair_mask_bgr.shape[0], hair_mask_bgr.shape[1]))
        for i in range(hair_mask_bgr.shape[0]):
            for j in range(hair_mask_bgr.shape[1]):
                if self.criterion(hair_mask_bgr[i, j]):
                    boolean_mask[i, j] = 1
                else:
                    boolean_mask[i, j] = 0

        return boolean_mask

    def process(self, image: np.ndarray) -> (int, int, int):
        """
        Parameters
        ----------
        image -- BGR np.ndarray

        Returns
        -------
            (r, g, b)
        """
        mask = self.get_boolean_mask(image)
        b, g, r = cv2.split(image)
        return int(np.median(r[mask == 1])), \
               int(np.median(g[mask == 1])), \
               int(np.median(b[mask == 1]))

    def apply_mask_on_image(self, image: np.ndarray) -> np.ndarray:
        mask = self.get_boolean_mask(image)
        annotated = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j]:
                    annotated[i, j] = [0, 255, 255]

        return annotated


def process():
    image = cv2.imread('1.jpg')
    hs = Hair(eps_criterion)
    print(hs.process(image))
    annotated = hs.apply_mask_on_image(image)
    cv2.imwrite('annotated.png', annotated)


if __name__ == '__main__':
    process()
