import numpy as np
import cv2


def _match_histograms(source, reference, alpha):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image with adjustable strength.

    Arguments:
    source -- 1D numpy array, source array to be transformed
    reference -- 1D numpy array, reference array to match the histogram of
    alpha -- float, strength of histogram matching (0.0: no change, 1.0: full matching)

    Returns:
    matched -- 1D numpy array, source array after histogram matching with strength applied
    """

    # Get the histogram and cumulative distribution function (CDF) of the source image
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # Get the histogram and cumulative distribution function (CDF) of the reference image
    r_values, r_counts = np.unique(reference, return_counts=True)
    r_quantiles = np.cumsum(r_counts).astype(np.float64)
    r_quantiles /= r_quantiles[-1]

    # Interpolate pixel values from the reference image to match the source image CDF
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    # Apply the strength factor
    matched_values = (1 - alpha) * s_values[bin_idx] + alpha * interp_r_values[bin_idx]

    return matched_values.astype(np.uint8)


def binarize_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def match_histograms_by_region(img_src, img_ref, alpha=0.7):
    img_src = np.array(img_src)
    img_ref = np.array(img_ref)

    # 黒、白領域は中央値に置換（もっといい方法があるかも）
    mask_black = (img_ref[:, :, 0] <= 70) & (img_ref[:, :, 1] <= 70) & (img_ref[:, :, 2] <= 70)
    mask_white = (img_ref[:, :, 0] >= 170) & (img_ref[:, :, 1] >= 170) & (img_ref[:, :, 2] >= 170)
    img_ref[mask_black | mask_white] = np.median(img_ref, axis=(0, 1))

    img_src_gray = binarize_image(img_src)
    img_ref_gray = binarize_image(img_ref)

    matched_image = np.zeros_like(img_src)

    for value in [0, 255]:
        mask1 = img_src_gray == value
        mask2 = img_ref_gray == value

        region1 = img_src[mask1]
        region2 = img_ref[mask2]

        for c in range(3):
            matched_image[:, :, c][mask1] = _match_histograms(region1[:, c], region2[:, c], alpha)

    return matched_image
