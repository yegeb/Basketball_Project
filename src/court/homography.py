import numpy as np
import cv2

class Homography:
    """
    A class that encapsulates the computation and application of a homography transformation
    from a set of source points to a set of target points in 2D space.

    Attributes:
        model (np.ndarray): The 3x3 homography matrix computed from source to target points.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initializes the Homography object by computing the transformation matrix.

        Args:
            source (np.ndarray): Array of source points with shape (N, 2).
            target (np.ndarray): Array of target points with shape (N, 2).

        Raises:
            ValueError: If the shapes of source and target do not match,
                        or if they are not 2D arrays of shape (N, 2),
                        or if the homography matrix could not be computed.
        """
        # Check if source and target arrays have the same shape
        if source.shape != target.shape:
            raise ValueError("Source and Target must have the same shape")
        
        # Ensure both arrays are 2D with shape (N, 2)
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("Source and Target must be 2D arrays of shape (N, 2)")

        # Convert to float32 for OpenCV compatibility
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Compute the homography matrix using OpenCV
        self.model, _ = cv2.findHomography(source, target)

        # Raise an error if homography computation failed
        if self.model is None:
            raise ValueError("Homography matrix is None")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Applies the computed homography transformation to a set of points.

        Args:
            points (np.ndarray): Array of input points with shape (N, 2).

        Returns:
            np.ndarray: Transformed points in the target space, shape (N, 2).

        Raises:
            ValueError: If input is not a 2D array of shape (N, 2).
        """
        # Return early if the input array is empty
        if points.size == 0:
            return points

        # Ensure input has proper shape
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a 2D array of shape (N, 2)")

        # Reshape to (N, 1, 2) for OpenCV, and ensure float32 type
        points = points.reshape(-1, 1, 2).astype(np.float32)

        # Apply homography transformation
        trans_points = cv2.perspectiveTransform(points, self.model)

        # Return reshaped result as (N, 2)
        return trans_points.reshape(-1, 2).astype(np.float32)