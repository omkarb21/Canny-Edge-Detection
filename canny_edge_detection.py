import numpy as np
from skimage.transform import warp_polar
from scipy import ndimage
from scipy.fft import fftshift, fft2
from skimage import io, color,filters
import matplotlib.pyplot as plt


def canny_edge_detect(sigma):
    # Loading input image and converting to grayscale
    image_path = "checkerboard.png"
    input_image = io.imread(image_path)
    if input_image.ndim == 3:  # Convert RGB to grayscale
        gray_image = color.rgb2gray(input_image)
    else:
        gray_image = input_image / 255.0 if input_image.max() > 1 else input_image

    # Step 1: Applying Gaussian Smoothing to the input grayscale image
    smoothed_result = filters.gaussian(gray_image, sigma=sigma)

    # Step 2: finding the Gradients using Sobel filter
    grad_x = ndimage.sobel(smoothed_result, axis=0)
    grad_y = ndimage.sobel(smoothed_result, axis=1)
    gradient_orientation = np.arctan2(grad_y, grad_x)
    gradient_magnitude = np.hypot(grad_x, grad_y)


    # Step 3: Applying Non-Maximum Suppression
    gradient_orientation_degrees = (np.rad2deg(gradient_orientation) + 180) % 180
    thinned_edges = non_max_suppression(gradient_magnitude, gradient_orientation_degrees)

    # Step 4: Performing Dual Thresholding
    lower_threshold = 0.5
    upper_threshold = 0.9
    strong_edge_pixels = thinned_edges > upper_threshold  # Pixels above the upper threshold are strong edges

    # Identifying weak edge pixels
    weak_edge_pixels = (thinned_edges >= lower_threshold) & (thinned_edges <= upper_threshold)

    # Creating final edge map
    edge_map_result = np.zeros_like(thinned_edges, dtype=np.float32)
    edge_map_result[strong_edge_pixels] = 0.9
    edge_map_result[weak_edge_pixels] = 1.2

    inverted_image = 1 - edge_map_result
    # Display the results
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(inverted_image, cmap='gray')
    plt.title(f"Canny Edge Detection with sigma={sigma}, Low_t={lower_threshold}, High_t={upper_threshold}")
    plt.axis('off')


    plt.show()

def non_max_suppression(grad_magnitude, grad_orientation):
    edge_thinning = np.zeros_like(grad_magnitude)
    rows, cols = grad_magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Determine the direction of the gradient
            angle = grad_orientation[i, j]
            neighbor1, neighbor2 = 0, 0

            #Checking for horizonatal edges
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbor1 = grad_magnitude[i, j + 1]
                neighbor2 = grad_magnitude[i, j - 1]
            #Checking for Diagonal edges
            elif 22.5 <= angle < 67.5:
                neighbor1 = grad_magnitude[i + 1, j - 1]
                neighbor2 = grad_magnitude[i - 1, j + 1]
            #Checking for vertical edges
            elif 67.5 <= angle < 112.5:
                neighbor1 = grad_magnitude[i + 1, j]
                neighbor2 = grad_magnitude[i - 1, j]
            #checking for diagonal edges
            elif 112.5 <= angle < 157.5:
                neighbor1 = grad_magnitude[i - 1, j - 1]
                neighbor2 = grad_magnitude[i + 1, j + 1]

            # Keep the pixel if it is greater than both neighboring pixels
            if grad_magnitude[i, j] >= neighbor1 and grad_magnitude[i, j] >= neighbor2:
                edge_thinning[i, j] = grad_magnitude[i, j]

    return edge_thinning

if __name__ == "__main__":
    #q1_code()

    sigma_values = [0.1, 0.2, 0.3, 0.5, 1.0, 1.2, 1.4,1.5, 1.8, 2.0, 2.5]
    #sigma_values=[1.5]
    for sigma in sigma_values:
        canny_edge_detect(sigma)