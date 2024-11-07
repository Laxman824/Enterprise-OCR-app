import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import streamlit as st

class ImagePreprocessor:
    @staticmethod
    def enhance_image(image: Image.Image, 
                     enhance_contrast: bool = True,
                     remove_noise: bool = True,
                     deskew: bool = True) -> Image.Image:
        """
        Enhance image quality for better OCR results
        """
        # Convert PIL Image to numpy array
        img_np = np.array(image)

        # Convert to grayscale if image is colored
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        if remove_noise:
            # Denoise image
            gray = cv2.fastNlMeansDenoising(gray)

        if enhance_contrast:
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

        if deskew:
            # Deskew image if needed
            gray = ImagePreprocessor._deskew(gray)

        # Binarization
        binary = cv2.adaptiveThreshold(
            gray, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return Image.fromarray(binary)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """
        Deskew the image using contour detection
        """
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        
        if angle < -45:
            angle = 90 + angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, 
            M, 
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated

class DocumentScanner:
    @staticmethod
    def scan_document(image: Image.Image) -> Optional[Image.Image]:
        """
        Find document boundaries and crop/transform accordingly
        """
        # Convert to numpy array
        img_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blur, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find largest contour
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we found a rectangle
        if len(approx) == 4:
            # Get corners in order
            corners = DocumentScanner._order_points(approx.reshape(4, 2))
            
            # Get target dimensions
            width = max([
                np.linalg.norm(corners[0] - corners[1]),
                np.linalg.norm(corners[2] - corners[3])
            ])
            height = max([
                np.linalg.norm(corners[0] - corners[3]),
                np.linalg.norm(corners[1] - corners[2])
            ])
            
            # Create target points
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Get transformation matrix
            M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
            
            # Warp image
            warped = cv2.warpPerspective(img_np, M, (int(width), int(height)))
            
            return Image.fromarray(warped)
            
        return None

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect

def optimize_image_size(image: Image.Image, 
                       max_size: int = 1800) -> Image.Image:
    """
    Optimize image size while maintaining aspect ratio
    """
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        image = image.resize((new_width, new_height), Image.Reductive)
    
    return image