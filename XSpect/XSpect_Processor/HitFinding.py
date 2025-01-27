import numpy as np

def basic_detect(images, cutoff_multiplier=1.0, absolute_threshold=None):
    """
    Detect hits in a batch of images based on a dynamic or absolute threshold.

    Parameters:
    images (numpy.ndarray): 3D array of images with shape (n_images, height, width).
    cutoff_multiplier (float): The multiplier for the standard deviation to determine the dynamic threshold.
    absolute_threshold (float, optional): An absolute threshold for deciding hits. If provided, this will be used instead of the dynamic threshold.

    Returns:
    numpy.ndarray: Indices of the images that are considered hits.
    """
    sum_images = np.sum(images, axis=(1, 2))

    if absolute_threshold is not None:
        threshold = absolute_threshold
    else:
        mean_sum = np.median(sum_images)
        std_sum = np.std(sum_images)
        threshold = mean_sum - cutoff_multiplier * std_sum
    hits = sum_images > threshold
    hit_indices = np.where(hits)[0]
    return hit_indices