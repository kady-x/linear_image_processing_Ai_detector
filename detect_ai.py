import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_image_linear_algebra(image_path):
    # 1. Load Image (Matrix Construction)
    # -----------------------------------
    # Read as Color (BGR) and Grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Prepare the figure for the paper
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Linear Algebra Analysis: {image_path}", fontsize=16)

    # 2. Covariance Matrix of Color Channels
    # --------------------------------------
    # Reshape image to a list of pixels: Matrix X (N x 3)
    pixels = img_rgb.reshape(-1, 3)
    
    # Calculate Covariance Matrix: C = (X-mu)^T * (X-mu) / (N-1)
    # We use NumPy's built-in function for stability
    cov_matrix = np.cov(pixels, rowvar=False)
    
    # Plotting the Correlation Heatmap
    plt.subplot(2, 3, 1)
    plt.imshow(cov_matrix, cmap='viridis', interpolation='nearest')
    plt.title("Color Covariance Matrix")
    plt.colorbar()
    plt.xticks([0, 1, 2], ['R', 'G', 'B'])
    plt.yticks([0, 1, 2], ['R', 'G', 'B'])
    
    # Analysis Note:
    # Real images usually have high covariance between channels.
    # Some AI artifacts cause "independent" noise, lowering these values.

    # 3. Singular Value Decomposition (SVD)
    # -------------------------------------
    # Decompose the Grayscale Matrix A = U * S * Vt
    U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
    
    # Plot Singular Values (The Spectrum)
    plt.subplot(2, 3, 2)
    plt.plot(np.log10(S), color='purple', linewidth=2)
    plt.title("Singular Value Spectrum (Log Scale)")
    plt.xlabel("Index")
    plt.ylabel("Log(Singular Value)")
    plt.grid(True)
    
    # Analysis Note:
    # Look for sudden drops or "steps" in the tail of this curve.
    # Real camera noise usually creates a smooth "long tail."

    # 4. SVD Reconstruction Error (Low Rank Approximation)
    # ----------------------------------------------------
    # Reconstruct using only top 20 singular values (Rank-20 Approx)
    k = 20
    reconst_img = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k, :])
    
    # Calculate Residuals (Noise/Artifacts) = Original - Reconstructed
    residuals = img_gray - reconst_img
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(residuals), cmap='hot')
    plt.title(f"Residuals (Rank-{k} approx)")
    plt.axis('off')

    # Analysis Note:
    # In AI images, these residuals sometimes show a "Grid" or "Checkerboard".

    # 5. Principal Component Analysis (PCA)
    # -------------------------------------
    # We treat each row of the image as a data point
    pca = PCA(n_components=20)
    pca.fit(img_gray)
    
    # Plot Cumulative Variance Explained
    plt.subplot(2, 3, 4)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='green')
    plt.title("PCA: Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Ratio")
    plt.grid(True)

    # 6. Show Original Image
    plt.subplot(2, 3, 5)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print Numerical Data for the Paper
    print("-" * 30)
    print("NUMERICAL RESULTS FOR PAPER")
    print("-" * 30)
    print("Covariance Matrix (R, G, B):\n", cov_matrix)
    print("\nTop 5 Singular Values:\n", S[:5])
    print("-" * 30)

analyze_image_linear_algebra('imgs/sunset_real.jpg')
# analyze_image_linear_algebra('imgs/sunset_nanobanana_ai.png')