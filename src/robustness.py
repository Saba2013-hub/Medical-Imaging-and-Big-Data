
"""Controlled degradations applied to processed 512-px images; background is kept black."""
import cv2, numpy as np

def _keep_mask(src, out):
    out[src.max(axis=2) == 0] = 0
    return out

def gaussian_noise(img, sigma, seed):
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return _keep_mask(img, np.clip(out, 0, 255).astype(np.uint8))

def gaussian_blur(img, sigma):
    return _keep_mask(img, cv2.GaussianBlur(img, (0, 0), sigma))

def gamma(img, g):
    lut = np.clip(255 * (np.arange(256) / 255) ** g, 0, 255).astype(np.uint8)
    return _keep_mask(img, cv2.LUT(img, lut))

def jpeg(img, quality):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return _keep_mask(img, cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

def noise_then_median(img, sigma, seed, k=3):
    return _keep_mask(img, cv2.medianBlur(gaussian_noise(img, sigma, seed), k))
