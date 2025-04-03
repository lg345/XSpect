from scipy.optimize import curve_fit
import numpy as np
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def stdev_to_fwhm_gaussian(stddev):
    # FWHM for a Gaussian is 2.3548 * standard deviation
   # return 2 * np.sqrt(2 * np.log(2)) * stddev
    return  2.3548 * stddev

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

def gaussian_fwhm(x, y, x1, x2,all_vals= False):
    """
    Fit a Gaussian to the data within a specified x range and calculate the FWHM of the peak.

    Parameters:
    x (np.ndarray): Array of x (energy) values.
    y (np.ndarray): Array of y (intensity) values.
    x1 (float): The starting position in x that defines the range to fit the Gaussian.
    x2 (float): The ending position in x that defines the range to fit the Gaussian.

    Returns:
    float: The FWHM of the fitted Gaussian peak.
    """
    # Ensure x1 is less than x2
    if x1 > x2:
        x1, x2 = x2, x1

    # Select the data points within the specified range
    mask = (x >= x1) & (x <= x2)
    x_range = x[mask]
    y_range = y[mask]
    
    if len(x_range) == 0 or len(y_range) == 0:
        raise ValueError("No data points found within the specified range.")

    # Initial guess for the Gaussian parameters: amplitude, mean, stddev
    initial_guess = [np.max(y_range), np.mean(x_range), np.std(x_range)]

    # Fit the Gaussian
    try:
        popt, pcov = curve_fit(gaussian, x_range, y_range, p0=initial_guess)
        amplitude, mean, stddev = popt
        fwhm = stdev_to_fwhm_gaussian(stddev)
        if all_vals:
            return popt
        else:
            return fwhm
    except RuntimeError:
        raise RuntimeError("Gaussian fit did not converge.")