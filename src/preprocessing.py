
"""Preprocessing used for modelling.
Version A = the stored 512-px images from Cell 4a (crop, pad, area-resize, black background).
Version B = version A + CLAHE on the L channel of LAB (fixed before any model is trained)."""
import cv2, numpy as np

CLAHE_CLIP, CLAHE_TILES = 2.0, (8, 8)

def load_processed(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def retina_mask(img):
    return img.max(axis=2) > 0

def version_a(img):
    return img

def version_b(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    out[~retina_mask(img)] = 0
    return out

def ben_graham(img, sigma=10):
    """High-pass filter (image minus Gaussian blur). Illustration only, not used for modelling."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    out[~retina_mask(img)] = 0
    return out
