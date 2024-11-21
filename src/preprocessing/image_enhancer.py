# src/preprocessing/image_enhancer.py
import cv2
import numpy as np
from PIL import Image
import io


class ImageEnhancer:
    def enhance_image(self, img_data: bytes) -> Image.Image:
        """Enhanced image preprocessing with improved error handling."""
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_data))

            # Convert to grayscale and resize in one step
            width, height = img.size
            img_gray = img.convert('L').resize(
                (width * 2, height * 2),
                Image.Resampling.LANCZOS
            )

            # Convert to numpy array for OpenCV processing
            img_array = np.array(img_gray)

            # Apply optimized image processing pipeline
            denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
            thresh = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            return Image.fromarray(thresh)

        except Exception as e:
            print(f"Image enhancement failed: {type(e).__name__}: {str(e)}")
            # Return original image as fallback
            return Image.open(io.BytesIO(img_data))
