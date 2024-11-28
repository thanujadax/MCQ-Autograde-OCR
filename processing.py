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
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast adjustment
        adjusted = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=0)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        return thresh
    
    def detect_bubbles(self, preprocessed: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        for contour in contours:
            # Filter by area and circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.7 and area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                bubbles.append((x, y, w, h))
        
        return bubbles
    
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
        preprocessed = self.preprocess_image(image)
        debug_images['preprocessed'] = preprocessed
        
        # Detect bubbles
        bubbles = self.detect_bubbles(preprocessed)
        
        # Draw bubbles on debug image
        bubble_visualization = image.copy()
        for x, y, w, h in bubbles:
            cv2.rectangle(bubble_visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
        debug_images['detected_bubbles'] = bubble_visualization
        
        # Analyze answers
        answers, confidence = self.analyze_answers(preprocessed, bubbles)
        
        return ProcessingResult(answers, confidence, bubbles), debug_images
