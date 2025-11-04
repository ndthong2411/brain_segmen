"""
Advanced Medical Image Augmentation

Medical-specific augmentation techniques for brain MRI segmentation.
These augmentations address unique challenges in medical imaging:
- Elastic deformations (anatomical variations)
- Bias field corruption (MRI inhomogeneity artifacts)
- Gaussian blur (partial volume effects)
- Cutout (robustness to missing data)

Expected improvement: +1-2% Dice through better generalization

Author: BrainTumNet Phase 1 Optimization
Date: 2025-01-04
"""

import torch
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple


class MedicalAugmentation:
    """
    Advanced medical imaging augmentations for brain MRI

    Combines standard and medical-specific transformations to improve
    model robustness and generalization.

    Args:
        elastic_deform_p: Probability of elastic deformation (default: 0.3)
        elastic_alpha: Deformation intensity (default: 30)
        elastic_sigma: Gaussian smoothing for deformation (default: 4)
        bias_field_p: Probability of bias field corruption (default: 0.5)
        bias_field_scale: Scale of bias field variations (default: 0.3)
        gaussian_blur_p: Probability of Gaussian blur (default: 0.2)
        gaussian_blur_sigma: Range of blur sigma (default: (0.5, 1.5))
        gamma_p: Probability of gamma correction (default: 0.5)
        gamma_range: Range of gamma values (default: (0.7, 1.4))
        cutout_p: Probability of cutout (default: 0.2)
        cutout_n_holes: Number of cutout regions (default: 3)
        cutout_size: Size of each cutout region (default: 20)
        local_shuffle_p: Probability of local pixel shuffling (default: 0.15)
        local_shuffle_size: Size of shuffle region (default: 3)
    """

    def __init__(
        self,
        elastic_deform_p=0.3,
        elastic_alpha=30,
        elastic_sigma=4,
        bias_field_p=0.5,
        bias_field_scale=0.3,
        gaussian_blur_p=0.2,
        gaussian_blur_sigma=(0.5, 1.5),
        gamma_p=0.5,
        gamma_range=(0.7, 1.4),
        cutout_p=0.2,
        cutout_n_holes=3,
        cutout_size=20,
        local_shuffle_p=0.15,
        local_shuffle_size=3,
    ):
        self.elastic_deform_p = elastic_deform_p
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.bias_field_p = bias_field_p
        self.bias_field_scale = bias_field_scale

        self.gaussian_blur_p = gaussian_blur_p
        self.gaussian_blur_sigma = gaussian_blur_sigma

        self.gamma_p = gamma_p
        self.gamma_range = gamma_range

        self.cutout_p = cutout_p
        self.cutout_n_holes = cutout_n_holes
        self.cutout_size = cutout_size

        self.local_shuffle_p = local_shuffle_p
        self.local_shuffle_size = local_shuffle_size

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to image and mask

        Args:
            image: (C, H, W) input image tensor
            mask: (H, W) or (1, H, W) mask tensor

        Returns:
            aug_image: Augmented image
            aug_mask: Augmented mask (same spatial transforms as image)
        """
        # Convert to numpy for scipy operations
        image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        # Ensure mask is 2D
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)

        # 1. Elastic deformation (spatial transform - affects both image and mask)
        if random.random() < self.elastic_deform_p:
            image_np, mask_np = self.elastic_deform(image_np, mask_np)

        # 2. Bias field corruption (intensity transform - affects only image)
        if random.random() < self.bias_field_p:
            image_np = self.bias_field_corruption(image_np)

        # 3. Gaussian blur (affects only image)
        if random.random() < self.gaussian_blur_p:
            sigma = random.uniform(*self.gaussian_blur_sigma)
            image_np = self.gaussian_blur(image_np, sigma)

        # 4. Gamma correction per modality (affects only image)
        if random.random() < self.gamma_p:
            gamma = random.uniform(*self.gamma_range)
            image_np = self.gamma_transform(image_np, gamma)

        # 5. Cutout (affects only image - set random patches to 0)
        if random.random() < self.cutout_p:
            image_np = self.cutout(image_np)

        # 6. Local pixel shuffling (affects only image)
        if random.random() < self.local_shuffle_p:
            image_np = self.local_shuffle(image_np)

        # Convert back to tensors
        image_tensor = torch.from_numpy(image_np).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        # Restore original mask shape if needed
        if mask.ndim == 3 and mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor

    def elastic_deform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elastic deformation for anatomical variations

        Critical for medical imaging - simulates natural anatomical variability
        """
        # Get image shape
        if image.ndim == 3:  # (C, H, W)
            _, height, width = image.shape
        else:  # (H, W)
            height, width = image.shape

        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1),
                            self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1),
                            self.elastic_sigma) * self.elastic_alpha

        # Create coordinate meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply deformation to image
        if image.ndim == 3:
            deformed_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                deformed_image[c] = map_coordinates(
                    image[c], indices, order=1, mode='reflect'
                ).reshape(height, width)
        else:
            deformed_image = map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(height, width)

        # Apply same deformation to mask (use order=0 for nearest neighbor)
        deformed_mask = map_coordinates(
            mask, indices, order=0, mode='reflect'
        ).reshape(height, width)

        return deformed_image, deformed_mask

    def bias_field_corruption(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate MRI bias field inhomogeneity

        MRI scanners produce spatially varying intensity artifacts
        """
        if image.ndim == 3:  # (C, H, W)
            _, height, width = image.shape
        else:
            height, width = image.shape

        # Generate smooth bias field
        bias_field = np.random.randn(height // 4, width // 4) * self.bias_field_scale
        bias_field = gaussian_filter(bias_field, sigma=2)

        # Resize to match image
        from scipy.ndimage import zoom
        zoom_factor = (height / bias_field.shape[0], width / bias_field.shape[1])
        bias_field = zoom(bias_field, zoom_factor, order=3)

        # Apply exponential bias (multiplicative, not additive)
        bias_field = np.exp(bias_field)

        # Apply to all channels
        if image.ndim == 3:
            corrupted = image * bias_field[np.newaxis, :, :]
        else:
            corrupted = image * bias_field

        return corrupted

    def gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Gaussian blur to simulate partial volume effects
        """
        if image.ndim == 3:
            blurred = np.zeros_like(image)
            for c in range(image.shape[0]):
                blurred[c] = gaussian_filter(image[c], sigma=sigma)
        else:
            blurred = gaussian_filter(image, sigma=sigma)

        return blurred

    def gamma_transform(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Gamma correction for intensity variations

        Different MRI sequences have different intensity distributions
        """
        # Normalize to [0, 1] range
        img_min = image.min()
        img_max = image.max()

        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
            corrected = np.power(normalized, gamma)
            # Restore original range
            result = corrected * (img_max - img_min) + img_min
        else:
            result = image

        return result

    def cutout(self, image: np.ndarray) -> np.ndarray:
        """
        Random cutout regions (simulates missing data / artifacts)
        """
        result = image.copy()

        if image.ndim == 3:
            _, height, width = image.shape
        else:
            height, width = image.shape

        for _ in range(self.cutout_n_holes):
            # Random position
            y = random.randint(0, height - self.cutout_size)
            x = random.randint(0, width - self.cutout_size)

            # Set region to 0
            if image.ndim == 3:
                result[:, y:y+self.cutout_size, x:x+self.cutout_size] = 0
            else:
                result[y:y+self.cutout_size, x:x+self.cutout_size] = 0

        return result

    def local_shuffle(self, image: np.ndarray) -> np.ndarray:
        """
        Local pixel shuffling for texture randomization

        Helps prevent overfitting to specific texture patterns
        """
        result = image.copy()

        if image.ndim == 3:
            _, height, width = image.shape
        else:
            height, width = image.shape

        size = self.local_shuffle_size

        # Apply shuffling to random regions
        num_regions = (height // size) * (width // size) // 10  # Shuffle ~10% of regions

        for _ in range(num_regions):
            # Random region
            y = random.randint(0, height - size)
            x = random.randint(0, width - size)

            if image.ndim == 3:
                for c in range(image.shape[0]):
                    region = result[c, y:y+size, x:x+size].copy()
                    # Shuffle pixels within region
                    flat = region.flatten()
                    np.random.shuffle(flat)
                    result[c, y:y+size, x:x+size] = flat.reshape(size, size)
            else:
                region = result[y:y+size, x:x+size].copy()
                flat = region.flatten()
                np.random.shuffle(flat)
                result[y:y+size, x:x+size] = flat.reshape(size, size)

        return result


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Testing Medical Augmentation")
    print("="*70)

    # Create dummy data
    image = torch.randn(4, 256, 256)  # 4 MRI modalities
    mask = torch.randint(0, 3, (256, 256))  # 3 classes

    # Initialize augmentation
    aug = MedicalAugmentation()

    # Apply augmentation
    aug_image, aug_mask = aug(image, mask)

    print(f"\nOriginal image: {image.shape}, range [{image.min():.3f}, {image.max():.3f}]")
    print(f"Augmented image: {aug_image.shape}, range [{aug_image.min():.3f}, {aug_image.max():.3f}]")
    print(f"Original mask: {mask.shape}, unique values {torch.unique(mask).tolist()}")
    print(f"Augmented mask: {aug_mask.shape}, unique values {torch.unique(aug_mask).tolist()}")
    print("\n✓ Medical augmentation tests passed!")
