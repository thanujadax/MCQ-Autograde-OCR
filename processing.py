import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class ProcessingResult:
    answers: Dict[int, str]
    confidence: Dict[int, float]
    detected_bubbles: List[Tuple[int, int, int, int]]

class MCQProcessor:
    def __init__(self, threshold: int = 128, bubble_size: int = 20, contrast: float = 1.0):
        self.threshold = threshold
        self.bubble_size = bubble_size
        self.contrast = contrast
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast adjustment
        adjusted = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=0)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # C constant
        )
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Apply morphological operations
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        debug_images = {
            'grayscale': gray,
            'contrast_adjusted': adjusted,
            'blurred': blurred,
            'threshold': thresh,
            'cleaned': cleaned
        }
        
        return cleaned, debug_images
    
    def detect_bubbles(self, preprocessed: np.ndarray, original_image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        # Find contours
        contours, hierarchy = cv2.findContours(
            preprocessed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        # Create visualization image
        vis_image = original_image.copy()
        
        min_area = np.pi * (self.bubble_size / 2) ** 2 * 0.5
        max_area = np.pi * (self.bubble_size * 1.5) ** 2
        
        for contour in contours:
            # Filter by area and circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Improved filtering criteria
            if (circularity > 0.8 and  # More strict circularity
                min_area < area < max_area and  # Dynamic area based on bubble_size
                cv2.contourArea(contour) / (cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1]) > 0.7):  # Rectangularity check
                
                x, y, w, h = cv2.boundingRect(contour)
                bubbles.append((x, y, w, h))
                
                # Draw both marked and unmarked bubbles
                # Blue for detected bubbles
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Check if bubble is marked (filled)
                roi = preprocessed[y:y+h, x:x+w]
                if np.mean(roi) > 127:  # If bubble is filled
                    # Draw filled circle inside the rectangle
                    center = (x + w//2, y + h//2)
                    radius = min(w, h) // 4
                    cv2.circle(vis_image, center, radius, (0, 0, 255), -1)
        
        return bubbles, vis_image
    
    def analyze_answers(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Tuple[Dict[int, str], Dict[int, float]]:
        answers = {}
        confidence = {}
        
        # Group bubbles by row
        bubbles.sort(key=lambda x: x[1])  # Sort by y coordinate
        rows = []
        current_row = []
        last_y = None
        
        for bubble in bubbles:
            if last_y is None or abs(bubble[1] - last_y) < 20:
                current_row.append(bubble)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [bubble]
            last_y = bubble[1]
            
        if current_row:
            rows.append(current_row)
        
        # Analyze each row
        for i, row in enumerate(rows):
            row.sort(key=lambda x: x[0])  # Sort by x coordinate
            max_darkness = 0
            selected_option = None
            
            for j, bubble in enumerate(row):
                x, y, w, h = bubble
                roi = image[y:y+h, x:x+w]
                darkness = 255 - np.mean(roi)
                
                if darkness > max_darkness:
                    max_darkness = darkness
                    selected_option = chr(65 + j)  # Convert to A, B, C, etc.
            
            answers[i + 1] = selected_option
            confidence[i + 1] = max_darkness / 255
        
        return answers, confidence
    
    def process(self, image: np.ndarray) -> Tuple[ProcessingResult, Dict[str, np.ndarray]]:
        # Create debug images dictionary
        debug_images = {
            'original': image.copy()
        }
        
        # Preprocess
        preprocessed, preprocess_debug = self.preprocess_image(image)
        debug_images.update(preprocess_debug)
        
        # Detect bubbles
        bubbles, bubble_visualization = self.detect_bubbles(preprocessed, image)
        debug_images['detected_bubbles'] = bubble_visualization
        
        # Analyze answers
        answers, confidence = self.analyze_answers(preprocessed, bubbles)
        
        return ProcessingResult(answers, confidence, bubbles), debug_images
