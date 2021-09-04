import numpy as np
import math
import cv2


def calculate_psnr(img1, img2):
    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_psnr(img1[i], img2[i])
        return temp

    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 0.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *
                                                            (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_ssim(img1[i], img2[i])
        return temp

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions!')


def calculate_rmse(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions!')

    # multiple images
    if img1.ndim == 4:
        temp = 0
        for i in range(len(img1)):
            temp += calculate_rmse(img1[i], img2[i])
        return temp

    num = (img1 - img2) ** 2
    denom = img2 ** 2 + 1.0e-2
    relative_mse = np.divide(num, denom)
    relative_mse_mean = 0.5 * np.mean(relative_mse)
    return relative_mse_mean
