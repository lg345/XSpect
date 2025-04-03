import numpy as np
from numpy.polynomial import Polynomial

def exclude_regions(x, y, regions):
    """
    Exclude specific regions from the data.

    Parameters:
    x (np.ndarray): Array of x values.
    y (np.ndarray): Array of y values.
    regions (list of tuples): List of (x_start, x_end) pairs defining the regions to be excluded.

    Returns:
    np.ndarray, np.ndarray: x and y values with the specified regions excluded.
    """
    mask = np.ones(len(x), dtype=bool)
    for (x_start, x_end) in regions:
        mask &= ~((x >= x_start) & (x <= x_end))
    return x[mask], y[mask]

def polynomial_subtraction(x, y, degree, exclude=None, return_coefficients=False):
    """
    Fit and subtract a polynomial of specified degree from the y data, excluding specified regions.

    Parameters:
    x (np.ndarray): Array of x values.
    y (np.ndarray): Array of y values.
    degree (int): Degree of the polynomial to fit and subtract.
    exclude (list of tuples, optional): List of (x_start, x_end) pairs defining the regions to be excluded from fitting.
    return_coefficients (bool, optional): Flag to return the coefficients of the fitted polynomial.

    Returns:
    np.ndarray: The y values after polynomial subtraction.
    np.ndarray (optional): The coefficients of the fitted polynomial, if return_coefficients is True.
    """
    if exclude:
        x_fit, y_fit = exclude_regions(x, y, exclude)
    else:
        x_fit, y_fit = x, y

    # Fit the polynomial
    coeffs = np.polyfit(x_fit, y_fit, degree)
    poly = np.poly1d(coeffs)
    
    # Subtract the polynomial from the original y data
    y_subtracted = y - poly(x)
    
    if return_coefficients:
        return y_subtracted, coeffs
    else:
        return y_subtracted