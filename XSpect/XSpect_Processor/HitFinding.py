import numpy as np

def basic_detect(images, cutoff_multiplier=None, absolute_threshold=None):
    """
    Detect hits in a batch of images based on a dynamic or absolute threshold.

    Parameters:
    images (numpy.ndarray): 3D array of images with shape (n_images, height, width).
    cutoff_multiplier (float): The multiplier for the standard deviation to determine the dynamic threshold.
    absolute_threshold (float, optional): An absolute threshold for deciding hits. If provided, this will be used in conjunction with the dynamic threshold.

    Returns:
    numpy.ndarray: Indices of the images that are considered hits.
    """
    sum_images = np.sum(images, axis=(1, 2))
    
    if absolute_threshold is not None:
        absolute_hits = sum_images > absolute_threshold
        filtered_sum_images = sum_images[absolute_hits]
    else:
        absolute_hits = np.ones_like(sum_images, dtype=bool)  # all True
        filtered_sum_images = sum_images

    if cutoff_multiplier is not None and filtered_sum_images.size > 0:
        mean_sum = np.median(filtered_sum_images)
        std_sum = np.std(filtered_sum_images)
        dynamic_threshold = mean_sum - cutoff_multiplier * std_sum
        dynamic_hits = sum_images > dynamic_threshold
    else:
        mean_sum = None
        std_sum = None
        dynamic_threshold = None
        dynamic_hits = np.ones_like(sum_images, dtype=bool)  # all True

    # Combine both conditions with AND
    hits = absolute_hits & dynamic_hits

    # Get the indices where hits are True
    hit_indices = np.where(hits)[0]

    return hit_indices, mean_sum, std_sum, dynamic_threshold, sum_images

# Example usage:
# images = np.random.rand(5, 100, 100)  # 5 images of 100x100 pixels
# hit_indices, mean_sum, std_sum, threshold, sum_images = basic_detect(images, cutoff_multiplier=2.5, absolute_threshold=1000)