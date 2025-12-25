import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_gradient_covariance(image_path):
    # 1. Load and Preprocess
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return

    # Resize for consistency (Standard 512x512)
    img_bgr = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to Grayscale for Gradient Analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Gradient Covariance Analysis: {image_path}", fontsize=14)

    # 2. Calculate Gradients (The Math from your Image)
    # Sobel operators calculate the derivative in X and Y directions
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # Gx: Horizontal changes
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # Gy: Vertical changes

    # Calculate Gradient Magnitude (for visualization)
    # This creates the "Sketch" look from your reference image
    magnitude = cv2.magnitude(gx, gy)

    # 3. Compute Gradient Covariance Matrix
    # Flatten the gradients into vectors
    gx_flat = gx.flatten()
    gy_flat = gy.flatten()
    
    # Stack them: Matrix M where rows are pixels, columns are [Gx, Gy]
    gradients = np.vstack((gx_flat, gy_flat))
    
    # Covariance Matrix C = (M * M.T) / N
    cov_matrix = np.cov(gradients)
    
    # Eigenvalues help us see if noise is "stretched" in one direction
    eigenvalues, _ = np.linalg.eig(cov_matrix)

    # 4. Visualization

    # Plot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Plot 2: Gradient Magnitude (The "Structure")
    plt.subplot(1, 3, 2)
    # Normalize to visible range 0-1 for plotting
    plt.imshow(magnitude, cmap='gray', vmin=0, vmax=np.percentile(magnitude, 95))
    plt.title("Gradient Map (Edges)")
    plt.axis('off')

    # Plot 3: Covariance Heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
    plt.title("Gradient Covariance\n(Gx vs Gy)")
    
    # Annotate the heatmap with values
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cov_matrix[i, j]:.1f}", 
                     ha="center", va="center", color="black", fontsize=12)
            
    plt.xticks([0, 1], ['Gx', 'Gy'])
    plt.yticks([0, 1], ['Gx', 'Gy'])

    plt.tight_layout()
    plt.show()

    # Print Text Results for Paper
    print("-" * 30)
    print(f"RESULTS FOR: {image_path}")
    print("-" * 30)
    print("Gradient Covariance Matrix:")
    print(cov_matrix)
    print("\nEigenvalues (Spread of Gradients):")
    print(eigenvalues)
    print("-" * 30)

# analyze_gradient_covariance('imgs/sunset_real.jpg')
# analyze_gradient_covariance('imgs/real1.jpg')
# analyze_gradient_covariance('imgs/sunset_nanobanana_ai.png')
analyze_gradient_covariance('imgs/nanobanana_ai1.png')




# ====== Other Linear Algebra Analyses ======

def analyze_image_linear_algebra(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Linear Algebra Analysis: {image_path}", fontsize=16)

    pixels = img_rgb.reshape(-1, 3)

    cov_matrix = np.cov(pixels, rowvar=False)
    
    plt.subplot(2, 3, 1)
    plt.imshow(cov_matrix, cmap='viridis', interpolation='nearest')
    plt.title("Color Covariance Matrix")
    plt.colorbar()
    plt.xticks([0, 1, 2], ['R', 'G', 'B'])
    plt.yticks([0, 1, 2], ['R', 'G', 'B'])
    
    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
    
    plt.subplot(2, 3, 2)
    plt.plot(np.log10(S), color='purple', linewidth=2)
    plt.title("Singular Value Spectrum (Log Scale)")
    plt.xlabel("Index")
    plt.ylabel("Log(Singular Value)")
    plt.grid(True)
    
    k = 20
    reconst_img = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k, :])
    
    residuals = img_gray - reconst_img
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(residuals), cmap='hot')
    plt.title(f"Residuals (Rank-{k} approx)")
    plt.axis('off')

    pca = PCA(n_components=20)
    pca.fit(img_gray)
    
    plt.subplot(2, 3, 4)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='green')
    plt.title("PCA: Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Ratio")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("-" * 30)
    print("NUMERICAL RESULTS FOR PAPER")
    print("-" * 30)
    print("Covariance Matrix (R, G, B):\n", cov_matrix)
    print("\nTop 5 Singular Values:\n", S[:5])
    print("-" * 30)


def analyze_image_linear_algebra_refined(image_path, label="Image"):
    target_size = (512, 512)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return

    img_bgr = cv2.resize(img_bgr, target_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Analysis: {label} (512x512)", fontsize=16)

    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)

    plt.subplot(1, 3, 1)
    plt.plot(np.log10(S), color='purple', linewidth=2)
    plt.title("SVD Spectrum (Log Scale)")
    plt.xlabel("Index")
    plt.ylabel("Log(Singular Value)")
    plt.grid(True)

    k = 30
    reconst_img = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k, :])
    residuals = np.abs(img_gray - reconst_img)
    
    residuals_norm = cv2.normalize(residuals, None, 0, 255, cv2.NORM_MINMAX)
    
    plt.subplot(1, 3, 2)
    plt.imshow(residuals_norm, cmap='inferno')
    plt.title(f"Enhanced Residuals (Rank-{k})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_rgb)
    plt.title("Original (Resized)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()