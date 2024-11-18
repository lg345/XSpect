import numpy as np
from scipy.ndimage import rotate
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class XSpectDetectorProcessor:
    """
    Processor for 2D detector data to perform image normalization, edge detection, 
    and alignment of signals.
    """
    
    def __init__(self, image):
        """
        Initialize the processor with the image.

        Parameters
        ----------
        image : ndarray
            2D numpy array of the detector data (e.g., keV measurements).
        """
        self.image = image
        self.image_8bit = self.normalize_to_8bit(image)
        self.edges = None
        self.aligned_image = None
    
    def normalize_to_8bit(self, image):
        """
        Normalize image data to 8-bit range [0, 255].

        Parameters
        ----------
        image : ndarray
            Input image data to be normalized.

        Returns
        -------
        ndarray
            Normalized 8-bit image.
        """
        min_value = np.min(image)
        max_value = np.max(image)

        # Normalize the image to range [0, 255]
        normalized_image = (image - min_value) / (max_value - min_value) * 255

        # Convert to 8-bit values
        image_8bit = normalized_image.astype(np.uint8)
        
        return image_8bit

    def detect_edges(self, low_threshold=30, high_threshold=100):
        """
        Detect edges in the image using the Canny edge detector.

        Parameters
        ----------
        low_threshold : int, optional
            Lower threshold for the hysteresis procedure (default is 50).
        high_threshold : int, optional
            Upper threshold for the hysteresis procedure (default is 150).

        Returns
        -------
        ndarray
            Binary image with detected edges.
        """
        self.edges = cv2.Canny(self.image_8bit, low_threshold, high_threshold)
        return self.edges


    def find_optimal_rotation_angle(self):
        """
        Calculate the optimal rotation angle to align the signals.

        Returns
        -------
        float
            Optimal rotation angle in degrees.
        
        Raises
        ------
        ValueError
            If edges have not been detected.
        """
        if self.edges is None:
            raise ValueError("Edges have not been detected. Call `detect_edges` first.")
        
        # Find all non-zero (signal) pixels
        non_zero_coords = np.column_stack(np.nonzero(self.edges))
        
        #Identifying multiple XES edges.
        clustering = DBSCAN(eps=5, min_samples=10).fit(non_zero_coords)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        angles = []
        
        for label in unique_labels:
            if label == -1:
                # -1 indicates noise
                continue
            
            cluster_coords = non_zero_coords[labels == label]
            
            mean = np.mean(cluster_coords, axis=0)
            centered_coords = cluster_coords - mean
            covariance_matrix = np.cov(centered_coords, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            
           
            principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
            
     
            angle = np.arctan2(principal_vector[1], principal_vector[0])
            
           
            angle_degrees = np.degrees(angle)
            angles.append(angle_degrees)

        
        optimal_angle = np.mean(angles)
        self.angles=angles
        
        if optimal_angle > 90:
            optimal_angle -= 180
        elif optimal_angle < -90:
            optimal_angle += 180
        
        return optimal_angle

    def align_image(self):
        """
        Align the image using the calculated optimal rotation angle.

        Returns
        -------
        ndarray
            Aligned image.
        """
        optimal_angle = self.find_optimal_rotation_angle()
        self.aligned_image = rotate(self.image, -optimal_angle, reshape=False)
        return self.aligned_image
    
    def plot_images(self):
        """
        Plot the original image, edge-detected image, and aligned image.

        Raises
        ------
        ValueError
            If edges or aligned image are not available.
        """
        if self.edges is None:
            raise ValueError("Edges have not been detected. Call `detect_edges` first.")
        if self.aligned_image is None:
            raise ValueError("Image has not been aligned. Call `align_image` first.")
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 3),dpi=100)
        
        # Original image
        axes[0].imshow(self.image, cmap='RdBu')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Edge-detected image
        axes[1].imshow(self.edges, cmap='gray')
        axes[1].set_title("XES Detected Image")
        axes[1].axis('off')
        
        # Aligned image
        axes[2].imshow(self.aligned_image, cmap='RdBu')
        axes[2].set_title("Aligned Image")
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()