import numpy as np
import matplotlib.pyplot as plt

# Simulate a 3D image with 2 channels (shape: Batch x Channels x Depth x Height x Width)
batch_size = 256
height = 4
width = 4
channels = 2
np.random.seed(42)

# Random 3D image data (4 images, 2 channels, 4x4x4 pixels)
images = np.random.randn(
    batch_size, channels, height, width
)  # Random pixel values between 0 and 10
images[:, 0, :, :] = images[:, 0, :, :] * 2
images[:, 1, :, :] = images[:, 1, :, :] * 10


# Batch Normalization function for 3D images (across batch, depth, height, and width)
def batch_normalization_3d(images):
    # Compute mean and variance for each channel across the batch, depth, height, and width
    mean = np.mean(images, axis=(1, 2, 3), keepdims=True)
    variance = np.var(images, axis=(1, 2, 3), keepdims=True)

    # Normalize the images
    normalized_images = (images - mean) / np.sqrt(
        variance + 1e-5
    )  # Adding epsilon to avoid division by zero
    return normalized_images, mean, variance


# Apply batch normalization
normalized_images, mean, variance = batch_normalization_3d(images)
plt.figure(figsize=(16, 6))
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(
    images[:, 0].flatten(),
    images[:, 1].flatten(),
    alpha=0.2,
    s=30,
)
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.set_xlabel("Channel 1 Values")
ax1.set_ylabel("Channel 2 Values")
ax1.grid()
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(
    normalized_images[:, 0].flatten(),
    normalized_images[:, 1].flatten(),
    alpha=0.2,
    s=30,
)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel("Channel 1 Values")
ax2.set_ylabel("Channel 2 Values")
ax2.grid(True)
plt.savefig("comparison.jpg")
