import numpy as np

def first_moment(x, y, x1, x2):
    """
    Calculate the first moment (center of mass) of a peak within a specified x range.

    Parameters:
    x (np.ndarray): Array of x (energy) values.
    y (np.ndarray): Array of y (intensity) values.
    x1 (float): The starting position in x that defines the range to calculate the first moment.
    x2 (float): The ending position in x that defines the range to calculate the first moment.

    Returns:
    float: The first moment (center of mass) of the peak within the specified range.
    """
    # Ensure x1 is less than x2
    if x1 > x2:
        x1, x2 = x2, x1

    # Select the data points within the specified range
    mask = (x >= x1) & (x <= x2)
    x_range = x[mask]
    y_range = y[mask]
    
    # Calculate the first moment (center of mass)
    numerator = np.sum(x_range * y_range)
    denominator = np.sum(y_range)
    
    if denominator == 0:
        raise ValueError("The sum of y values within the specified range is zero, cannot calculate first moment.")
    
    first_moment_value = numerator / denominator
    
    return first_moment_value
