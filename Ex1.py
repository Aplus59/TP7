import cv2 as cv
import sys
from matplotlib import pyplot as plt
import math
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from scipy.stats import entropy as scipy_entropy

def quantize_grayscale(window, levels=32):
    window = window.astype(np.uint8)
    if levels <= 1:
        return np.zeros_like(window, dtype=np.uint8)
    q = (window.astype(np.uint16) * levels) // 256
    q[q == levels] = levels - 1
    return q.astype(np.uint8)

def compute_glcm(window, levels=32, dx=1, dy=0):
    window_q = quantize_grayscale(window, levels=levels)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    height, width = window_q.shape

    # support dx, dy >= 0 only
    max_row = height - max(0, dy)
    max_col = width - max(0, dx)

    for y in range(max_row):
        for x in range(max_col):
            i = int(window_q[y, x])
            j = int(window_q[y + dy, x + dx])
            glcm[i, j] += 1

    return glcm

def normalize_glcm(glcm):
    total = np.sum(glcm)
    if total == 0:
        return glcm
    return glcm / total

def compute_texture_features(p):
    levels = p.shape[0]
    i, j = np.indices((levels, levels))

    # scale indices back to 0..255 range so thresholds remain meaningful
    scale = 256.0 / levels
    I = i * scale
    J = j * scale

    mu = np.sum(I * p)
    variance = np.sum((I - mu) ** 2 * p)
    contrast = np.sum((I - J) ** 2 * p)

    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))

    return {
        'variance': variance,
        'contrast': contrast,
        'entropy': entropy
    }

def compute_sliding_window_features(image, window_size=32, step=16, levels=32, dx=1, dy=0):
    height, width = image.shape
    num_rows = (height - window_size) // step + 1
    num_cols = (width - window_size) // step + 1

    variance_map = np.zeros((num_rows, num_cols), dtype=np.float64)
    contrast_map = np.zeros((num_rows, num_cols), dtype=np.float64)
    entropy_map = np.zeros((num_rows, num_cols), dtype=np.float64)

    for iy, y in enumerate(range(0, height - window_size + 1, step)):
        for ix, x in enumerate(range(0, width - window_size + 1, step)):
            window = image[y:y + window_size, x:x + window_size]
            glcm = compute_glcm(window, levels=levels, dx=dx, dy=dy)
            p = normalize_glcm(glcm)
            feats = compute_texture_features(p)
            variance_map[iy, ix] = feats['variance']
            contrast_map[iy, ix] = feats['contrast']
            entropy_map[iy, ix] = feats['entropy']

    return variance_map, contrast_map, entropy_map

def threshold_features(variance_map, contrast_map, entropy_map, var_thresh=50, cont_thresh=100, ent_thresh=4):
    mask = (contrast_map > cont_thresh) & (entropy_map > ent_thresh) & (variance_map > var_thresh)
    return mask.astype(np.uint8) * 255

def upscale_feature_map(feature_map, original_shape, window_size, step):
    height, width = original_shape
    out = np.zeros((height, width), dtype=feature_map.dtype)
    num_rows, num_cols = feature_map.shape

    for iy in range(num_rows):
        for ix in range(num_cols):
            y = iy * step
            x = ix * step
            out[y:y + window_size, x:x + window_size] = feature_map[iy, ix]

    return out

def first_order_texture(image, window_size=32, step=16):
    height, width = image.shape
    num_rows = (height - window_size) // step + 1
    num_cols = (width - window_size) // step + 1

    variance_map = np.zeros((num_rows, num_cols), dtype=np.float64)
    entropy_map = np.zeros((num_rows, num_cols), dtype=np.float64)

    for iy, y in enumerate(range(0, height - window_size + 1, step)):
        for ix, x in enumerate(range(0, width - window_size + 1, step)):
            window = image[y:y + window_size, x:x + window_size]
            variance_map[iy, ix] = np.var(window)
            hist, _ = np.histogram(window.flatten(), bins=256, range=(0, 256), density=True)
            entropy_map[iy, ix] = scipy_entropy(hist + 1e-10)

    return variance_map, entropy_map

def first_order_texture_builtin(image, radius=8):
    ent = sk_entropy(image, disk(radius)).astype(np.float64)
    var = cv.blur((image.astype(np.float64) ** 2), (2 * radius + 1, 2 * radius + 1)) - \
          (cv.blur(image.astype(np.float64), (2 * radius + 1, 2 * radius + 1)) ** 2)
    return var, ent

def lbp_texture(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp

def existing_glcm_implementation(image, window_size=32, step=16, levels=32, dx=1, dy=0):
    height, width = image.shape
    num_rows = (height - window_size) // step + 1
    num_cols = (width - window_size) // step + 1

    variance_map = np.zeros((num_rows, num_cols), dtype=np.float64)
    contrast_map = np.zeros((num_rows, num_cols), dtype=np.float64)
    entropy_map = np.zeros((num_rows, num_cols), dtype=np.float64)

    for iy, y in enumerate(range(0, height - window_size + 1, step)):
        for ix, x in enumerate(range(0, width - window_size + 1, step)):
            window = image[y:y + window_size, x:x + window_size]
            window_q = quantize_grayscale(window, levels=levels)

            g = graycomatrix(
                window_q,
                distances=[int(max(1, math.hypot(dx, dy)))],
                angles=[math.atan2(dy, dx) if not (dx == 0 and dy == 0) else 0.0],
                levels=levels,
                symmetric=False,
                normed=True
            )

            # contrast from library
            contrast_map[iy, ix] = float(graycoprops(g, 'contrast')[0, 0]) * ((256.0 / levels) ** 2)

            # variance + entropy computed on scaled indices to match our custom features
            p = g[:, :, 0, 0]
            i, j = np.indices((levels, levels))

            scale = 256.0 / levels
            I = i * scale
            J = j * scale

            mu = np.sum(I * p)
            variance_map[iy, ix] = np.sum((I - mu) ** 2 * p)

            p_nonzero = p[p > 0]
            entropy_map[iy, ix] = -np.sum(p_nonzero * np.log(p_nonzero))

    return variance_map, contrast_map, entropy_map

def visualize_results(original, glcm_mask, first_order_mask, lbp):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(glcm_mask, cmap='gray')
    axs[0, 1].set_title('GLCM Threshold Mask')
    axs[1, 0].imshow(first_order_mask, cmap='gray')
    axs[1, 0].set_title('First-Order Threshold Mask')
    axs[1, 1].imshow(lbp, cmap='gray')
    axs[1, 1].set_title('LBP Image')
    plt.show()

if __name__ == "__main__":
    img = cv.imread(r".\bin\zebra_1.tif", cv.IMREAD_GRAYSCALE)

    window_size = 32
    step = 16
    dx, dy = 1, 0
    levels = 32  # number of gray levels to quantize to (e.g., 8, 16, 32, 64)

    # 1) Own GLCM + sliding window maps (quantized + scaled back for thresholds)
    var_map, cont_map, ent_map = compute_sliding_window_features(
        img, window_size=window_size, step=step, levels=levels, dx=dx, dy=dy
    )
    glcm_mask_small = threshold_features(var_map, cont_map, ent_map, var_thresh=50, cont_thresh=100, ent_thresh=4)
    glcm_mask = upscale_feature_map(glcm_mask_small, img.shape, window_size, step).astype(np.uint8)

    # 3) First-order texture (manual) + threshold
    fo_var_map, fo_ent_map = first_order_texture(img, window_size=window_size, step=step)
    first_order_mask_small = ((fo_var_map > 200) & (fo_ent_map > 4)).astype(np.uint8) * 255
    first_order_mask = upscale_feature_map(first_order_mask_small, img.shape, window_size, step).astype(np.uint8)

    # 3) First-order texture (built-in entropy) 
    fo_var_builtin, fo_ent_builtin = first_order_texture_builtin(img, radius=8)

    # 4) Existing implementation (skimage graycomatrix/graycoprops) for comparison (quantized + scaled back)
    var_map_ref, cont_map_ref, ent_map_ref = existing_glcm_implementation(
        img, window_size=window_size, step=step, levels=levels, dx=dx, dy=dy
    )

    # LBP (extra)
    lbp = lbp_texture(img)

    visualize_results(img, glcm_mask, first_order_mask, lbp)
