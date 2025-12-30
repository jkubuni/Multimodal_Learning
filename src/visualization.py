import numpy as np
import matplotlib.pyplot as plt

def save_pcd(points, filename):
    """Save points to a PCD file (ASCII format)."""
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")

def convert_lidar_to_pcd(npy_path, az, ze, output_path):
    """
    Convert range image to PCD file.
    This is required for proper display of the LiDAR point clouds in FiftyOne.
    Args:
        npy_path: Path to the .npy file containing range image.
        az: 1D array of azimuth angles (in radians).
        ze: 1D array of zenith angles (in radians).
        output_path: Path to save the output .pcd file.
    """
    r = np.load(npy_path)
    
    # Create meshgrid for angles
    # We assume r[i, j] corresponds to ze[i] (rows) and az[j] (cols)
    ZE, AZ = np.meshgrid(ze, az, indexing='ij')
    
    # Convert to Cartesian coordinates
    # Using Pinhole/Tangent model which is common for synthetic datasets
    tan_az = np.tan(AZ)
    tan_ze = np.tan(ZE)
    
    # Assuming r is Euclidean distance (range)
    norm = np.sqrt(1 + tan_az**2 + tan_ze**2)
    z = r / norm
    
    x = z * tan_az
    y = z * tan_ze
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    save_pcd(points, output_path)

def plot_losses(train_losses, val_losses=None, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracies(accuracies, title="Validation Accuracy"):
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
