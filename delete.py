import numpy as np


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> int:
    if image.shape != kernel.shape:
        return None

    weighted_sum = 0
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            weighted_sum += image[row, col] * kernel[row, col]
    return weighted_sum


def convolve(image: np.ndarray, kernel: np.ndarray):
    kernel_size = kernel.shape[0]
    conv_mat    = np.empty((4, 4))
    kr, kc      = 0,0
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            sub_image = image[row: row + kernel_size, col:col + kernel_size]
            ws = apply_filter(sub_image, kernel)
            if kc == 3:
                kc = 0
                kr += 1

            if kr >= 3:
                break

            conv_mat[kr, kc] = ws
            kc += 1
        kr += 1
    return conv_mat


image = np.array([
    (3, 0, 1, 2, 7, 4),
    (1, 5, 8, 9, 3, 1),
    (2, 7, 2, 5, 1, 3),
    (0, 1, 3, 1, 7, 8),
    (4, 2, 1, 6, 2, 8),
    (2, 4, 5, 2, 3, 9)
]).reshape(6, 6)
kernel = np.array([
    (1, 0, -1),
    (1, 0, -1),
    (1, 0, -1)
])

convolve(image, kernel)
