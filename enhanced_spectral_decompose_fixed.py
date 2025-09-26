import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from skimage import color
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_image(image_path, target_size=None):
    """Load and preprocess image for spectral analysis"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Don't resize - keep original dimensions to match patch count
    if target_size is not None:
        img = cv2.resize(img, target_size)
    
    # Convert to grayscale for spectral analysis
    gray = color.rgb2gray(img)
    
    return img, gray

def calculate_patch_grid_from_image(image_shape, patch_size=16):
    """Calculate patch grid dimensions from image shape and patch size"""
    h, w = image_shape[:2]  # Handle both (H,W) and (H,W,C) shapes
    patch_h = h // patch_size
    patch_w = w // patch_size
    return (patch_h, patch_w)

def load_precomputed_eigendata(eig_path):
    """Load precomputed eigenvalues and eigenvectors from .pth file"""
    data = torch.load(eig_path)
    eigenvals = data['eigenvalues'].numpy()
    eigenvecs = data['eigenvectors'].numpy()
    
    print(f"Loaded from {eig_path}:")
    print(f"  Eigenvalues shape: {eigenvals.shape}")
    print(f"  Eigenvectors shape: {eigenvecs.shape}")
    
    # Sort by eigenvalue (should already be sorted, but just to be sure)
    idx = np.argsort(eigenvals)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[idx, :]  # Sort rows (eigenvectors), not columns!
    
    print(f"  After sorting - Eigenvectors shape: {eigenvecs.shape}")
    
    return eigenvals, eigenvecs

def analyze_frequency_bands(eigenvals, eigenvecs, image_shape, patch_grid, original_image=None, n_bands=3):
    """Analyze different frequency bands"""
    n_components = len(eigenvals)  # Number of eigenvectors (50)
    n_patches = eigenvecs.shape[1]  # Number of patches is the SECOND dimension (1024 or 961)
    
    print(f"Number of eigenvectors: {n_components}")
    print(f"Number of patches per eigenvector: {n_patches}")
    print(f"Expected patches from image: {patch_grid[0] * patch_grid[1]}")
    print(f"Eigenvector shape: {eigenvecs.shape}")
    
    # Verify patch count matches
    if patch_grid[0] * patch_grid[1] != n_patches:
        print(f"WARNING: Patch count mismatch! Expected {patch_grid[0] * patch_grid[1]}, got {n_patches}")
        # Recalculate patch grid from actual patch count
        if n_patches == 1024:  # 32x32
            patch_grid = (32, 32)
        elif n_patches == 961:  # 31x31
            patch_grid = (31, 31)
        else:
            # General case: find best rectangular fit
            approx_size = int(np.sqrt(n_patches))
            if approx_size * approx_size == n_patches:
                patch_grid = (approx_size, approx_size)
            else:
                # Find best rectangular fit
                best_diff = float('inf')
                best_grid = (approx_size, approx_size)
                for h in range(1, n_patches + 1):
                    if n_patches % h == 0:
                        w = n_patches // h
                        diff = abs(h - w)
                        if diff < best_diff:
                            best_diff = diff
                            best_grid = (h, w)
                patch_grid = best_grid
        print(f"Using recalculated grid: {patch_grid}")
    
    band_size = n_components // n_bands
    
    bands = {
        'low': (0, band_size),
        'mid': (band_size, 2 * band_size),
        'high': (2 * band_size, n_components)
    }
    
    # First, compute full reconstruction from all bands
    full_reconstruction_patches = np.zeros(n_patches)
    for i in range(n_components):
        full_reconstruction_patches += eigenvals[i] * eigenvecs[i, :]
    
    # Reshape full reconstruction to image size
    try:
        full_reconstruction_grid = full_reconstruction_patches.reshape(patch_grid)
    except ValueError as e:
        print(f"Reshape error for full reconstruction: {e}")
        target_size = patch_grid[0] * patch_grid[1]
        if n_patches > target_size:
            full_reconstruction_patches = full_reconstruction_patches[:target_size]
        else:
            full_reconstruction_patches = np.pad(full_reconstruction_patches, (0, target_size - n_patches), 'constant')
        full_reconstruction_grid = full_reconstruction_patches.reshape(patch_grid)
    
    # Interpolate full reconstruction to image size
    scale_h = image_shape[0] / patch_grid[0]
    scale_w = image_shape[1] / patch_grid[1]
    full_reconstruction = zoom(full_reconstruction_grid, (scale_h, scale_w), order=1)
    
    if full_reconstruction.shape != image_shape:
        full_reconstruction = cv2.resize(full_reconstruction, (image_shape[1], image_shape[0]))
    
    band_analysis = {}
    
    for band_name, (start, end) in bands.items():
        # Eigenvalue statistics for this band
        band_eigenvals = eigenvals[start:end]
        band_eigenvecs = eigenvecs[start:end, :]  # Shape: (band_size, n_patches)
        
        # Compute energy in this band
        energy = np.sum(band_eigenvals)
        
        # Compute reconstruction using only this band
        # Sum weighted eigenvectors for this band
        reconstruction_patches = np.zeros(n_patches)
        for i in range(start, min(end, n_components)):
            reconstruction_patches += eigenvals[i] * eigenvecs[i, :]  # Use i-th eigenvector (all patches)
        
        # Reshape to patch grid first, then interpolate to image size
        try:
            reconstruction_grid = reconstruction_patches.reshape(patch_grid)
        except ValueError as e:
            print(f"Reshape error for {band_name}: {e}")
            print(f"Trying to reshape {n_patches} elements into {patch_grid}")
            # Emergency fallback: truncate or pad
            target_size = patch_grid[0] * patch_grid[1]
            if n_patches > target_size:
                reconstruction_patches = reconstruction_patches[:target_size]
            else:
                reconstruction_patches = np.pad(reconstruction_patches, (0, target_size - n_patches), 'constant')
            reconstruction_grid = reconstruction_patches.reshape(patch_grid)
        
        # Interpolate from patch grid to full image resolution
        scale_h = image_shape[0] / patch_grid[0]
        scale_w = image_shape[1] / patch_grid[1]
        reconstruction = zoom(reconstruction_grid, (scale_h, scale_w), order=1)
        
        # Ensure correct output shape
        if reconstruction.shape != image_shape:
            reconstruction = cv2.resize(reconstruction, (image_shape[1], image_shape[0]))
        
        # Calculate what remains when this frequency band is REMOVED
        # residual = full_reconstruction - this_band_reconstruction
        residual_image = full_reconstruction - reconstruction
        
        # Normalize residual image to [0, 1] range for proper visualization
        residual_image_normalized = residual_image.copy()
        residual_min, residual_max = residual_image_normalized.min(), residual_image_normalized.max()
        if residual_max > residual_min:
            residual_image_normalized = (residual_image_normalized - residual_min) / (residual_max - residual_min)
        else:
            residual_image_normalized = np.zeros_like(residual_image_normalized)
        
        band_analysis[band_name] = {
            'eigenvals': band_eigenvals,
            'eigenvecs': band_eigenvecs,
            'energy': energy,
            'reconstruction': reconstruction,
            'residual_image': residual_image_normalized,  # What remains when this band is removed
            'reconstruction_patches': reconstruction_patches,
            'patch_grid': patch_grid,
            'mean_eigenval': np.mean(band_eigenvals),
            'std_eigenval': np.std(band_eigenvals)
        }
    
    # Add full reconstruction to the analysis
    full_recon_normalized = full_reconstruction.copy()
    full_min, full_max = full_recon_normalized.min(), full_recon_normalized.max()
    if full_max > full_min:
        full_recon_normalized = (full_recon_normalized - full_min) / (full_max - full_min)
    
    return band_analysis, full_recon_normalized

def visualize_spectral_analysis(real_img, gen_img, real_gray, gen_gray, 
                               real_eigenvals, gen_eigenvals, 
                               real_bands, gen_bands):
    """Create comprehensive visualization of spectral analysis"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Original images
    plt.subplot(4, 6, 1)
    plt.imshow(real_img)
    plt.title('Real Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(4, 6, 2)
    plt.imshow(gen_img)
    plt.title('Generated Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Grayscale versions
    plt.subplot(4, 6, 3)
    plt.imshow(real_gray, cmap='gray')
    plt.title('Real (Grayscale)', fontsize=12)
    plt.axis('off')
    
    plt.subplot(4, 6, 4)
    plt.imshow(gen_gray, cmap='gray')
    plt.title('Generated (Grayscale)', fontsize=12)
    plt.axis('off')
    
    # Eigenvalue spectra comparison
    plt.subplot(4, 6, 5)
    plt.semilogy(real_eigenvals, 'b-', label='Real', alpha=0.7, linewidth=2)
    plt.semilogy(gen_eigenvals, 'r--', label='Generated', alpha=0.7, linewidth=2)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Eigenvalue Spectra', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Eigenvalue difference
    plt.subplot(4, 6, 6)
    min_len = min(len(real_eigenvals), len(gen_eigenvals))
    diff = real_eigenvals[:min_len] - gen_eigenvals[:min_len]
    plt.plot(diff, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Difference (Real - Gen)')
    plt.title('Eigenvalue Differences', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Band reconstructions
    bands = ['low', 'mid', 'high']
    band_colors = ['blue', 'orange', 'red']
    
    for i, (band, color) in enumerate(zip(bands, band_colors)):
        # Real image band reconstruction
        plt.subplot(4, 6, 7 + i*2)
        plt.imshow(real_bands[band]['reconstruction'], cmap='gray') 
        plt.title(f'Real - {band.capitalize()} Freq', fontsize=11, color=color, fontweight='bold')
        plt.axis('off')
        
        # Generated image band reconstruction
        plt.subplot(4, 6, 8 + i*2)
        plt.imshow(gen_bands[band]['reconstruction'], cmap='gray')
        plt.title(f'Gen - {band.capitalize()} Freq', fontsize=11, color=color, fontweight='bold')
        plt.axis('off')
    
    # Band energy comparison
    plt.subplot(4, 6, 13)
    band_names = list(real_bands.keys())
    real_energies = [real_bands[band]['energy'] for band in band_names]
    gen_energies = [gen_bands[band]['energy'] for band in band_names]
    
    x = np.arange(len(band_names))
    width = 0.35
    
    plt.bar(x - width/2, real_energies, width, label='Real', color='blue', alpha=0.7)
    plt.bar(x + width/2, gen_energies, width, label='Generated', color='red', alpha=0.7)
    
    plt.xlabel('Frequency Band')
    plt.ylabel('Total Energy')
    plt.title('Energy Distribution by Band', fontsize=12, fontweight='bold')
    plt.xticks(x, [b.capitalize() for b in band_names])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Band statistics comparison
    plt.subplot(4, 6, 14)
    real_means = [real_bands[band]['mean_eigenval'] for band in band_names]
    gen_means = [gen_bands[band]['mean_eigenval'] for band in band_names]
    
    plt.bar(x - width/2, real_means, width, label='Real', color='blue', alpha=0.7)
    plt.bar(x + width/2, gen_means, width, label='Generated', color='red', alpha=0.7)
    
    plt.xlabel('Frequency Band')
    plt.ylabel('Mean Eigenvalue')
    plt.title('Mean Eigenvalue by Band', fontsize=12, fontweight='bold')
    plt.xticks(x, [b.capitalize() for b in band_names])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mid-frequency detailed analysis
    plt.subplot(4, 6, 15)
    mid_start = len(real_eigenvals) // 3
    mid_end = 2 * len(real_eigenvals) // 3
    
    plt.semilogy(real_eigenvals[mid_start:mid_end], 'b-', 
                label='Real Mid-Freq', linewidth=2)
    plt.semilogy(gen_eigenvals[mid_start:mid_end], 'r--', 
                label='Gen Mid-Freq', linewidth=2)
    plt.xlabel('Mid-Frequency Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Mid-Frequency Detail', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reconstruction error analysis
    plt.subplot(4, 6, 16)
    reconstruction_errors = []
    for band in band_names:
        real_recon = real_bands[band]['reconstruction']
        gen_recon = gen_bands[band]['reconstruction']
        
        # Resize to common size for comparison
        common_size = min(real_recon.shape[0], gen_recon.shape[0])
        real_recon_resized = cv2.resize(real_recon, (common_size, common_size))
        gen_recon_resized = cv2.resize(gen_recon, (common_size, common_size))
        
        # Normalize reconstructions for fair comparison
        real_recon_norm = (real_recon_resized - real_recon_resized.mean()) / (real_recon_resized.std() + 1e-10)
        gen_recon_norm = (gen_recon_resized - gen_recon_resized.mean()) / (gen_recon_resized.std() + 1e-10)
        error = np.mean((real_recon_norm - gen_recon_norm)**2)
        reconstruction_errors.append(error)
    
    plt.bar(range(len(band_names)), reconstruction_errors, 
           color=['blue', 'orange', 'red'], alpha=0.7)
    plt.xlabel('Frequency Band')
    plt.ylabel('Reconstruction MSE')
    plt.title('Reconstruction Differences', fontsize=12, fontweight='bold')
    plt.xticks(range(len(band_names)), [b.capitalize() for b in band_names])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    output_path = '/home/zechenli/DinoV3/TestImages/laplacian_spectral_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comprehensive analysis saved to: {output_path}")
    
    return reconstruction_errors

def save_individual_reconstructions(real_bands, gen_bands, output_dir='/home/zechenli/DinoV3/TestImages'):
    """Save individual band reconstructions as separate images"""
    import os
    
    bands = ['low', 'mid', 'high']
    for band in bands:
        # Create figure for this band comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Real reconstruction
        axes[0].imshow(real_bands[band]['reconstruction'])
        axes[0].set_title(f'Real Image - {band.capitalize()} Frequency', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Generated reconstruction
        axes[1].imshow(gen_bands[band]['reconstruction'])
        axes[1].set_title(f'Generated Image - {band.capitalize()} Frequency', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save individual band comparison
        band_path = os.path.join(output_dir, f'{band}_frequency_comparison.png')
        plt.savefig(band_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"{band.capitalize()}-frequency comparison saved to: {band_path}")

def save_eigenvalue_analysis(real_eigenvals, gen_eigenvals, output_dir='/home/zechenli/DinoV3/TestImages'):
    """Save detailed eigenvalue analysis plots"""
    import os
    
    # Eigenvalue spectra comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Full spectrum comparison
    axes[0, 0].semilogy(real_eigenvals, 'b-', label='Real', alpha=0.8, linewidth=2)
    axes[0, 0].semilogy(gen_eigenvals, 'r--', label='Generated', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Eigenvalue Index')
    axes[0, 0].set_ylabel('Eigenvalue (log scale)')
    axes[0, 0].set_title('Complete Eigenvalue Spectra', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Eigenvalue differences
    min_len = min(len(real_eigenvals), len(gen_eigenvals))
    diff = real_eigenvals[:min_len] - gen_eigenvals[:min_len]
    axes[0, 1].plot(diff, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('Eigenvalue Index')
    axes[0, 1].set_ylabel('Difference (Real - Generated)')
    axes[0, 1].set_title('Eigenvalue Differences', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mid-frequency detailed analysis
    mid_start = len(real_eigenvals) // 3
    mid_end = 2 * len(real_eigenvals) // 3
    
    axes[1, 0].semilogy(range(mid_start, mid_end), real_eigenvals[mid_start:mid_end], 
                       'b-', label='Real Mid-Freq', linewidth=2)
    axes[1, 0].semilogy(range(mid_start, mid_end), gen_eigenvals[mid_start:mid_end], 
                       'r--', label='Generated Mid-Freq', linewidth=2)
    axes[1, 0].set_xlabel('Eigenvalue Index')
    axes[1, 0].set_ylabel('Eigenvalue (log scale)')
    axes[1, 0].set_title('Mid-Frequency Range Detail', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative energy
    real_cumsum = np.cumsum(real_eigenvals) / np.sum(real_eigenvals)
    gen_cumsum = np.cumsum(gen_eigenvals) / np.sum(gen_eigenvals)
    
    axes[1, 1].plot(real_cumsum, 'b-', label='Real', linewidth=2)
    axes[1, 1].plot(gen_cumsum, 'r--', label='Generated', linewidth=2)
    axes[1, 1].set_xlabel('Eigenvalue Index')
    axes[1, 1].set_ylabel('Cumulative Energy Fraction')
    axes[1, 1].set_title('Cumulative Energy Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    eigenval_path = os.path.join(output_dir, 'eigenvalue_detailed_analysis.png')
    plt.savefig(eigenval_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Detailed eigenvalue analysis saved to: {eigenval_path}")

def save_statistical_analysis(real_bands, gen_bands, reconstruction_errors, output_dir='/home/zechenli/DinoV3/TestImages'):
    """Save statistical comparison plots"""
    import os
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    band_names = list(real_bands.keys())
    x = np.arange(len(band_names))
    width = 0.35
    
    # Energy distribution
    real_energies = [real_bands[band]['energy'] for band in band_names]
    gen_energies = [gen_bands[band]['energy'] for band in band_names]
    
    axes[0, 0].bar(x - width/2, real_energies, width, label='Real', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, gen_energies, width, label='Generated', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Frequency Band')
    axes[0, 0].set_ylabel('Total Energy')
    axes[0, 0].set_title('Energy Distribution by Frequency Band', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([b.capitalize() for b in band_names])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean eigenvalues
    real_means = [real_bands[band]['mean_eigenval'] for band in band_names]
    gen_means = [gen_bands[band]['mean_eigenval'] for band in band_names]
    
    axes[0, 1].bar(x - width/2, real_means, width, label='Real', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, gen_means, width, label='Generated', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Frequency Band')
    axes[0, 1].set_ylabel('Mean Eigenvalue')
    axes[0, 1].set_title('Mean Eigenvalue by Frequency Band', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([b.capitalize() for b in band_names])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction errors
    colors = ['blue', 'orange', 'red']
    bars = axes[1, 0].bar(range(len(band_names)), reconstruction_errors, 
                         color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Frequency Band')
    axes[1, 0].set_ylabel('Reconstruction MSE')
    axes[1, 0].set_title('Reconstruction Differences (MSE)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(band_names)))
    axes[1, 0].set_xticklabels([b.capitalize() for b in band_names])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight the maximum error
    max_idx = np.argmax(reconstruction_errors)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(3)
    
    # Energy ratios
    energy_ratios = [gen_energies[i] / real_energies[i] if real_energies[i] > 0 else 0 
                    for i in range(len(band_names))]
    
    bars2 = axes[1, 1].bar(range(len(band_names)), energy_ratios, 
                          color=colors, alpha=0.7)
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Equal Energy')
    axes[1, 1].set_xlabel('Frequency Band')
    axes[1, 1].set_ylabel('Energy Ratio (Gen/Real)')
    axes[1, 1].set_title('Energy Ratios by Frequency Band', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(range(len(band_names)))
    axes[1, 1].set_xticklabels([b.capitalize() for b in band_names])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    stats_path = os.path.join(output_dir, 'statistical_analysis.png')
    plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Statistical analysis saved to: {stats_path}")

def save_residual_analysis(real_bands, gen_bands, real_full_recon, gen_full_recon, output_dir='/home/zechenli/DinoV3/TestImages'):
    """Save residual analysis showing what remains when each frequency band is removed"""
    import os
    
    bands = ['low', 'mid', 'high']
    band_colors = ['blue', 'orange', 'red']
    
    # Create comprehensive residual visualization
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    
    for i, (band, color) in enumerate(zip(bands, band_colors)):
        # Column 1: Full reconstruction
        axes[i, 0].imshow(real_full_recon, cmap='gray')
        axes[i, 0].set_title(f'Real Full\nReconstruction', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 2: This frequency band only
        axes[i, 1].imshow(real_bands[band]['reconstruction'], cmap='gray')
        axes[i, 1].set_title(f'Real {band.capitalize()}-Freq\nOnly', fontsize=12, fontweight='bold', color=color)
        axes[i, 1].axis('off')
        
        # Column 3: What remains when this band is removed (Real)
        axes[i, 2].imshow(real_bands[band]['residual_image'], cmap='gray')
        axes[i, 2].set_title(f'Real WITHOUT\n{band.capitalize()}-Freq', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Column 4: Generated frequency band only
        axes[i, 3].imshow(gen_bands[band]['reconstruction'], cmap='gray')
        axes[i, 3].set_title(f'Gen {band.capitalize()}-Freq\nOnly', fontsize=12, fontweight='bold', color=color)
        axes[i, 3].axis('off')
        
        # Column 5: What remains when this band is removed (Generated)
        axes[i, 4].imshow(gen_bands[band]['residual_image'], cmap='gray')
        axes[i, 4].set_title(f'Gen WITHOUT\n{band.capitalize()}-Freq', fontsize=12, fontweight='bold')
        axes[i, 4].axis('off')
    
    plt.suptitle('Frequency Band Analysis: What Remains When Each Band is Removed', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    residual_path = os.path.join(output_dir, 'residual_analysis.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Residual analysis saved to: {residual_path}")

def calculate_frequency_evaluation_metrics(real_bands, gen_bands, real_gray, gen_gray):
    """Calculate comprehensive evaluation metrics for frequency differences"""
    
    metrics = {}
    bands = ['low', 'mid', 'high']
    
    for band in bands:
        band_metrics = {}
        
        # 1. Reconstruction similarity metrics
        real_recon = real_bands[band]['reconstruction']
        gen_recon = gen_bands[band]['reconstruction']
        
        # Resize to common size for fair comparison
        common_size = min(real_recon.shape[0], gen_recon.shape[0])
        real_recon_resized = cv2.resize(real_recon, (common_size, common_size))
        gen_recon_resized = cv2.resize(gen_recon, (common_size, common_size))
        
        # Normalize for fair comparison
        real_norm = (real_recon_resized - real_recon_resized.mean()) / (real_recon_resized.std() + 1e-8)
        gen_norm = (gen_recon_resized - gen_recon_resized.mean()) / (gen_recon_resized.std() + 1e-8)
        
        # Mean Squared Error
        band_metrics['mse'] = np.mean((real_norm - gen_norm)**2)
        
        # Structural Similarity Index (SSIM)
        from skimage.metrics import structural_similarity as ssim
        band_metrics['ssim'] = ssim(real_norm, gen_norm, data_range=real_norm.max() - real_norm.min())
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse_for_psnr = np.mean((real_norm - gen_norm)**2)
        if mse_for_psnr > 0:
            band_metrics['psnr'] = 20 * np.log10(4.0 / np.sqrt(mse_for_psnr))  # 4.0 is range after normalization
        else:
            band_metrics['psnr'] = float('inf')
        
        # 2. Eigenvalue distribution differences
        real_eigenvals = real_bands[band]['eigenvals']
        gen_eigenvals = gen_bands[band]['eigenvals']
        
        # Kolmogorov-Smirnov test for distribution similarity
        from scipy.stats import ks_2samp
        ks_stat, ks_pvalue = ks_2samp(real_eigenvals, gen_eigenvals)
        band_metrics['ks_statistic'] = ks_stat
        band_metrics['ks_pvalue'] = ks_pvalue
        
        # Energy ratio
        real_energy = real_bands[band]['energy']
        gen_energy = gen_bands[band]['energy']
        band_metrics['energy_ratio'] = gen_energy / real_energy if real_energy > 0 else float('inf')
        
        # Mean eigenvalue difference
        band_metrics['mean_eigenval_diff'] = np.abs(np.mean(gen_eigenvals) - np.mean(real_eigenvals))
        
        # 3. Residual analysis metrics
        if 'residual_image' in real_bands[band] and 'residual_image' in gen_bands[band]:
            real_residual = real_bands[band]['residual_image']
            gen_residual = gen_bands[band]['residual_image']
            
            # Resize residuals to common size for correlation calculation
            common_size = min(real_residual.shape[0], gen_residual.shape[0])
            real_residual_resized = cv2.resize(real_residual, (common_size, common_size))
            gen_residual_resized = cv2.resize(gen_residual, (common_size, common_size))
            
            # Residual correlation
            real_flat = real_residual_resized.flatten()
            gen_flat = gen_residual_resized.flatten()
            
            # Handle potential NaN values in correlation
            if (len(real_flat) > 0 and len(gen_flat) > 0 and 
                np.std(real_flat) > 1e-10 and np.std(gen_flat) > 1e-10 and
                not np.any(np.isnan(real_flat)) and not np.any(np.isnan(gen_flat))):
                try:
                    correlation = np.corrcoef(real_flat, gen_flat)[0, 1]
                    band_metrics['residual_correlation'] = correlation if not np.isnan(correlation) else 0.0
                except:
                    band_metrics['residual_correlation'] = 0.0
            else:
                band_metrics['residual_correlation'] = 0.0
            
            # Residual energy (how much information is lost) - use original sizes
            band_metrics['real_residual_energy'] = np.mean(real_residual**2)
            band_metrics['gen_residual_energy'] = np.mean(gen_residual**2)
            band_metrics['residual_energy_ratio'] = band_metrics['gen_residual_energy'] / band_metrics['real_residual_energy'] if band_metrics['real_residual_energy'] > 0 else float('inf')
        
        metrics[band] = band_metrics
    
    # Overall metrics
    overall_metrics = {}
    
    # Which frequency band has the largest difference?
    mse_values = [metrics[band]['mse'] for band in bands]
    max_diff_band = bands[np.argmax(mse_values)]
    overall_metrics['max_difference_band'] = max_diff_band
    overall_metrics['max_difference_mse'] = max(mse_values)
    
    # Average SSIM across all bands
    ssim_values = [metrics[band]['ssim'] for band in bands]
    overall_metrics['average_ssim'] = np.mean(ssim_values)
    
    # Energy distribution similarity
    real_energies = [real_bands[band]['energy'] for band in bands]
    gen_energies = [gen_bands[band]['energy'] for band in bands]
    real_energy_dist = np.array(real_energies) / np.sum(real_energies)
    gen_energy_dist = np.array(gen_energies) / np.sum(gen_energies)
    overall_metrics['energy_distribution_mse'] = np.mean((real_energy_dist - gen_energy_dist)**2)
    
    metrics['overall'] = overall_metrics
    
    return metrics

def save_evaluation_metrics_plot(metrics, output_dir='/home/zechenli/DinoV3/TestImages'):
    """Save comprehensive evaluation metrics visualization"""
    import os
    
    bands = ['low', 'mid', 'high']
    band_colors = ['blue', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MSE comparison
    mse_values = [metrics[band]['mse'] for band in bands]
    bars1 = axes[0, 0].bar(bands, mse_values, color=band_colors, alpha=0.7)
    axes[0, 0].set_title('Mean Squared Error\nby Frequency Band', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight max difference
    max_idx = np.argmax(mse_values)
    bars1[max_idx].set_edgecolor('black')
    bars1[max_idx].set_linewidth(3)
    
    # 2. SSIM comparison
    ssim_values = [metrics[band]['ssim'] for band in bands]
    bars2 = axes[0, 1].bar(bands, ssim_values, color=band_colors, alpha=0.7)
    axes[0, 1].set_title('Structural Similarity (SSIM)\nby Frequency Band', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PSNR comparison
    psnr_values = [metrics[band]['psnr'] for band in bands]
    psnr_values = [min(p, 50) for p in psnr_values]  # Cap for visualization
    bars3 = axes[0, 2].bar(bands, psnr_values, color=band_colors, alpha=0.7)
    axes[0, 2].set_title('Peak Signal-to-Noise Ratio\nby Frequency Band', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('PSNR (dB)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Energy ratios
    energy_ratios = [metrics[band]['energy_ratio'] for band in bands]
    bars4 = axes[1, 0].bar(bands, energy_ratios, color=band_colors, alpha=0.7)
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Equal Energy')
    axes[1, 0].set_title('Energy Ratio\n(Generated/Real)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Energy Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. KS test statistics
    ks_stats = [metrics[band]['ks_statistic'] for band in bands]
    bars5 = axes[1, 1].bar(bands, ks_stats, color=band_colors, alpha=0.7)
    axes[1, 1].set_title('Kolmogorov-Smirnov Statistic\n(Eigenvalue Distribution)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('KS Statistic')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Residual energy comparison
    if 'real_residual_energy' in metrics['low']:
        real_residual_energies = [metrics[band]['real_residual_energy'] for band in bands]
        gen_residual_energies = [metrics[band]['gen_residual_energy'] for band in bands]
        
        x = np.arange(len(bands))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, real_residual_energies, width, label='Real', color='blue', alpha=0.7)
        axes[1, 2].bar(x + width/2, gen_residual_energies, width, label='Generated', color='red', alpha=0.7)
        axes[1, 2].set_title('Residual Energy\n(Information Loss)', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Residual Energy')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(bands)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Frequency Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Evaluation metrics plot saved to: {metrics_path}")

def save_frequency_energy_plot(real_eigenvals, gen_eigenvals, output_dir='/home/zechenli/DinoV3/TestImages'):
    """Save frequency vs spectral energy plot for real and generated images"""
    import os
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Smooth the eigenvalue curves for better visualization
    from scipy import ndimage
    
    # Apply slight smoothing to reduce noise in the curves
    real_smooth = ndimage.gaussian_filter1d(real_eigenvals, sigma=1.0)
    gen_smooth = ndimage.gaussian_filter1d(gen_eigenvals, sigma=1.0)
    
    # Frequency indices (representing different frequency components)
    freq_indices = np.arange(len(real_eigenvals))
    
    # Plot 1: Real Image
    ax1.plot(freq_indices, real_smooth, 'b-', linewidth=2, alpha=0.8)
    ax1.fill_between(freq_indices, real_smooth, alpha=0.3, color='blue')
    ax1.set_xlabel('Frequency Index (Low → High)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spectral Energy (Eigenvalue)', fontsize=12, fontweight='bold')
    ax1.set_title('Real Image\nFrequency vs Spectral Energy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to better show the range
    
    # Add frequency band annotations
    n_components = len(real_eigenvals)
    low_end = n_components // 3
    mid_end = 2 * n_components // 3
    
    ax1.axvspan(0, low_end, alpha=0.1, color='green', label='Low Freq')
    ax1.axvspan(low_end, mid_end, alpha=0.1, color='orange', label='Mid Freq')
    ax1.axvspan(mid_end, n_components, alpha=0.1, color='red', label='High Freq')
    ax1.legend(loc='upper right')
    
    # Plot 2: Generated Image
    ax2.plot(freq_indices[:len(gen_eigenvals)], gen_smooth, 'r-', linewidth=2, alpha=0.8)
    ax2.fill_between(freq_indices[:len(gen_eigenvals)], gen_smooth, alpha=0.3, color='red')
    ax2.set_xlabel('Frequency Index (Low → High)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spectral Energy (Eigenvalue)', fontsize=12, fontweight='bold')
    ax2.set_title('Generated Image\nFrequency vs Spectral Energy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to better show the range
    
    # Add frequency band annotations for generated image
    gen_components = len(gen_eigenvals)
    gen_low_end = gen_components // 3
    gen_mid_end = 2 * gen_components // 3
    
    ax2.axvspan(0, gen_low_end, alpha=0.1, color='green', label='Low Freq')
    ax2.axvspan(gen_low_end, gen_mid_end, alpha=0.1, color='orange', label='Mid Freq')
    ax2.axvspan(gen_mid_end, gen_components, alpha=0.1, color='red', label='High Freq')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    freq_energy_path = os.path.join(output_dir, 'frequency_energy_comparison.png')
    plt.savefig(freq_energy_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Frequency vs Energy comparison saved to: {freq_energy_path}")
    
    # Also create an overlay comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot both curves on the same graph
    min_len = min(len(real_eigenvals), len(gen_eigenvals))
    freq_indices_common = np.arange(min_len)
    
    ax.plot(freq_indices_common, real_smooth[:min_len], 'b-', linewidth=3, 
           label='Real Image', alpha=0.8)
    ax.plot(freq_indices_common, gen_smooth[:min_len], 'r--', linewidth=3, 
           label='Generated Image', alpha=0.8)
    
    ax.fill_between(freq_indices_common, real_smooth[:min_len], alpha=0.2, color='blue')
    ax.fill_between(freq_indices_common, gen_smooth[:min_len], alpha=0.2, color='red')
    
    ax.set_xlabel('Frequency Index (Low → High)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Energy (Eigenvalue)', fontsize=14, fontweight='bold')
    ax.set_title('Frequency vs Spectral Energy\nReal vs Generated Image Comparison', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add frequency band annotations
    low_end = min_len // 3
    mid_end = 2 * min_len // 3
    
    ax.axvspan(0, low_end, alpha=0.1, color='green', label='Low Freq Band')
    ax.axvspan(low_end, mid_end, alpha=0.1, color='orange', label='Mid Freq Band')
    ax.axvspan(mid_end, min_len, alpha=0.1, color='red', label='High Freq Band')
    
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    overlay_path = os.path.join(output_dir, 'frequency_energy_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Frequency vs Energy overlay comparison saved to: {overlay_path}")

def main():
    # File paths
    real_img_path = '/home/zechenli/DinoV3/TestImages/real.png'
    gen_img_path = '/home/zechenli/DinoV3/TestImages/gen0.png'
    output_dir = '/home/zechenli/DinoV3/TestImages'
    
    # Paths to precomputed eigenvalue/eigenvector files
    eig_paths = [
        "/home/zechenli/DinoV3/deep-spectral-segmentation/extract/data/MYDATA/eigs/laplacian/gen0.pth",
        "/home/zechenli/DinoV3/deep-spectral-segmentation/extract/data/MYDATA/eigs/laplacian/real.pth"
    ]
    
    print("Loading and preprocessing images...")
    
    # Load images without resizing to preserve original patch structure
    real_img, real_gray = load_and_preprocess_image(real_img_path)
    gen_img, gen_gray = load_and_preprocess_image(gen_img_path)
    
    print(f"Real image shape: {real_img.shape}")
    print(f"Generated image shape: {gen_img.shape}")
    
    print("Loading precomputed eigenvalues and eigenvectors...")
    
    # Load precomputed eigendata
    gen_eigenvals, gen_eigenvecs = load_precomputed_eigendata(eig_paths[0])  # gen0.pth
    real_eigenvals, real_eigenvecs = load_precomputed_eigendata(eig_paths[1])  # real.pth
    
    # Calculate patch grids from actual image dimensions
    real_patch_grid = calculate_patch_grid_from_image(real_gray.shape, patch_size=16)
    gen_patch_grid = calculate_patch_grid_from_image(gen_gray.shape, patch_size=16)
    
    print(f"Real patch grid: {real_patch_grid} = {real_patch_grid[0] * real_patch_grid[1]} patches")
    print(f"Gen patch grid: {gen_patch_grid} = {gen_patch_grid[0] * gen_patch_grid[1]} patches")
    
    print("Analyzing frequency bands...")
    
    # Analyze frequency bands with correct patch grids and original images for residual analysis
    real_bands, real_full_recon = analyze_frequency_bands(real_eigenvals, real_eigenvecs, real_gray.shape, real_patch_grid, original_image=real_gray)
    gen_bands, gen_full_recon = analyze_frequency_bands(gen_eigenvals, gen_eigenvecs, gen_gray.shape, gen_patch_grid, original_image=gen_gray)
    
    print("Creating and saving visualizations...")
    
    # Create and save main comprehensive visualization
    errors = visualize_spectral_analysis(real_img, gen_img, real_gray, gen_gray,
                                       real_eigenvals, gen_eigenvals,
                                       real_bands, gen_bands)
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_frequency_evaluation_metrics(real_bands, gen_bands, real_gray, gen_gray)
    
    # Save additional detailed analyses
    save_individual_reconstructions(real_bands, gen_bands, output_dir)
    save_eigenvalue_analysis(real_eigenvals, gen_eigenvals, output_dir)
    save_statistical_analysis(real_bands, gen_bands, errors, output_dir)
    save_frequency_energy_plot(real_eigenvals, gen_eigenvals, output_dir)
    
    # Save new residual and evaluation visualizations
    save_residual_analysis(real_bands, gen_bands, real_full_recon, gen_full_recon, output_dir)
    save_evaluation_metrics_plot(metrics, output_dir)
    
    # Print quantitative analysis
    print("\n" + "="*60)
    print("QUANTITATIVE ANALYSIS")
    print("="*60)
    
    print(f"\nEigenvalue ranges:")
    print(f"Real: min={real_eigenvals.min():.6f}, max={real_eigenvals.max():.6f}")
    print(f"Generated: min={gen_eigenvals.min():.6f}, max={gen_eigenvals.max():.6f}")
    
    print("\nBand Energy Comparison:")
    for band in ['low', 'mid', 'high']:
        real_energy = real_bands[band]['energy']
        gen_energy = gen_bands[band]['energy']
        ratio = gen_energy / real_energy if real_energy > 0 else float('inf')
        print(f"{band.capitalize()}-freq: Real={real_energy:.4f}, Gen={gen_energy:.4f}, Ratio={ratio:.4f}")
    
    print("\nBand Mean Eigenvalue Comparison:")
    for band in ['low', 'mid', 'high']:
        real_mean = real_bands[band]['mean_eigenval']
        gen_mean = gen_bands[band]['mean_eigenval']
        diff = gen_mean - real_mean
        print(f"{band.capitalize()}-freq: Real={real_mean:.6f}, Gen={gen_mean:.6f}, Diff={diff:.6f}")
    
    print("\nReconstruction Differences (MSE):")
    band_names = ['low', 'mid', 'high']
    for i, band in enumerate(band_names):
        print(f"{band.capitalize()}-freq reconstruction MSE: {errors[i]:.6f}")
    
    # Check if mid-frequency shows the largest difference (supporting the paper's claim)
    mid_error_idx = band_names.index('mid')
    if errors[mid_error_idx] == max(errors):
        print(f"\n✓ FINDING CONFIRMED: Mid-frequency shows largest reconstruction difference!")
        print(f"  This supports the paper's claim that diffusion models struggle with mid-frequency patterns.")
    else:
        print(f"\n⚠ Mid-frequency difference: {errors[mid_error_idx]:.6f}")
        print(f"  Largest difference in: {band_names[np.argmax(errors)]}-frequency")
    
    print(f"\nTotal number of eigenvalues processed:")
    print(f"Real: {len(real_eigenvals)}")
    print(f"Generated: {len(gen_eigenvals)}")
    
    # Print comprehensive evaluation metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*60)
    
    print(f"\nOverall Analysis:")
    print(f"Band with maximum difference: {metrics['overall']['max_difference_band']}")
    print(f"Maximum difference MSE: {metrics['overall']['max_difference_mse']:.6f}")
    print(f"Average SSIM across all bands: {metrics['overall']['average_ssim']:.4f}")
    print(f"Energy distribution MSE: {metrics['overall']['energy_distribution_mse']:.6f}")
    
    print(f"\nDetailed Metrics by Frequency Band:")
    bands = ['low', 'mid', 'high']
    for band in bands:
        print(f"\n{band.upper()}-FREQUENCY BAND:")
        print(f"  MSE: {metrics[band]['mse']:.6f}")
        print(f"  SSIM: {metrics[band]['ssim']:.4f}")
        print(f"  PSNR: {metrics[band]['psnr']:.2f} dB")
        print(f"  Energy Ratio (Gen/Real): {metrics[band]['energy_ratio']:.4f}")
        print(f"  KS Statistic: {metrics[band]['ks_statistic']:.4f}")
        print(f"  KS p-value: {metrics[band]['ks_pvalue']:.6f}")
        print(f"  Mean Eigenvalue Difference: {metrics[band]['mean_eigenval_diff']:.6f}")
        
        if 'residual_correlation' in metrics[band]:
            print(f"  Residual Correlation: {metrics[band]['residual_correlation']:.4f}")
            print(f"  Real Residual Energy: {metrics[band]['real_residual_energy']:.6f}")
            print(f"  Generated Residual Energy: {metrics[band]['gen_residual_energy']:.6f}")
            print(f"  Residual Energy Ratio: {metrics[band]['residual_energy_ratio']:.4f}")
    
    # Analysis summary
    print(f"\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Check which metrics support the mid-frequency hypothesis
    mse_values = [metrics[band]['mse'] for band in bands]
    ssim_values = [metrics[band]['ssim'] for band in bands]
    
    max_mse_band = bands[np.argmax(mse_values)]
    min_ssim_band = bands[np.argmin(ssim_values)]
    
    if max_mse_band == 'mid':
        print(f"✓ MSE ANALYSIS: Mid-frequency shows highest MSE ({metrics['mid']['mse']:.6f})")
        print(f"  Supporting the hypothesis that diffusion models struggle with mid-frequency patterns.")
    else:
        print(f"⚠ MSE ANALYSIS: {max_mse_band}-frequency shows highest MSE ({max(mse_values):.6f})")
    
    if min_ssim_band == 'mid':
        print(f"✓ SSIM ANALYSIS: Mid-frequency shows lowest SSIM ({metrics['mid']['ssim']:.4f})")
        print(f"  Supporting the hypothesis that diffusion models struggle with mid-frequency patterns.")
    else:
        print(f"⚠ SSIM ANALYSIS: {min_ssim_band}-frequency shows lowest SSIM ({min(ssim_values):.4f})")
    
    # Energy distribution analysis
    energy_ratios = [metrics[band]['energy_ratio'] for band in bands]
    energy_deviations = [abs(ratio - 1.0) for ratio in energy_ratios]
    max_energy_dev_band = bands[np.argmax(energy_deviations)]
    
    print(f"\nENERGY DISTRIBUTION:")
    for i, band in enumerate(bands):
        deviation = abs(energy_ratios[i] - 1.0)
        print(f"  {band.capitalize()}-freq energy ratio: {energy_ratios[i]:.4f} (deviation from 1.0: {deviation:.4f})")
    
    if max_energy_dev_band == 'mid':
        print(f"✓ Mid-frequency shows largest energy distribution difference")
    
    print(f"\nFiles saved to: {output_dir}")
    print(f"- laplacian_spectral_analysis.png (main comprehensive analysis)")
    print(f"- residual_analysis.png (UPDATED: proper residual visualization)")
    print(f"- evaluation_metrics.png (comprehensive metrics)")
    print(f"- frequency_energy_comparison.png")
    print(f"- Individual frequency band comparisons")
    print(f"- Statistical and eigenvalue analyses")

if __name__ == "__main__":
    main()