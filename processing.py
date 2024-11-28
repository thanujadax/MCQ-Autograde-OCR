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
    # Constants for answer sheet structure
    TOTAL_QUESTIONS = 90
    OPTIONS_PER_QUESTION = 5
    NUM_COLUMNS = 3
    QUESTIONS_PER_COLUMN = 30

    def __init__(self, threshold: int = 128, min_bubble_size: int = 15, max_bubble_size: int = 30, contrast: float = 1.0, correct_answers: Dict[int, str] = None):
        self.threshold = threshold
        self.min_bubble_size = min_bubble_size
        self.max_bubble_size = max_bubble_size
        self.contrast = contrast
        self.correct_answers = correct_answers or {}
    
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
    
    def split_into_columns(self, bubbles: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
        """Split bubbles into columns based on x-coordinates"""
        if not bubbles:
            return []
            
        # Sort bubbles by x-coordinate
        bubbles_sorted = sorted(bubbles, key=lambda x: x[0])
        
        # Find column boundaries using x-coordinates
        x_coords = [b[0] for b in bubbles_sorted]
        x_min, x_max = min(x_coords), max(x_coords)
        column_width = (x_max - x_min) / self.NUM_COLUMNS
        
        # Split into columns
        columns = [[] for _ in range(self.NUM_COLUMNS)]
        for bubble in bubbles_sorted:
            col_index = int((bubble[0] - x_min) / column_width)
            col_index = min(col_index, self.NUM_COLUMNS - 1)  # Ensure within bounds
            columns[col_index].append(bubble)
            
        return columns

    def organize_column_bubbles(self, column_bubbles: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
        """Organize bubbles in a column into rows of 5 options"""
        # Sort by y-coordinate
        sorted_bubbles = sorted(column_bubbles, key=lambda x: x[1])
        
        # Group into rows based on y-coordinate proximity
        rows = []
        current_row = []
        last_y = None
        
        for bubble in sorted_bubbles:
            if last_y is None or abs(bubble[1] - last_y) < self.min_bubble_size:
                current_row.append(bubble)
            else:
                if len(current_row) == self.OPTIONS_PER_QUESTION:
                    # Sort row by x-coordinate
                    current_row.sort(key=lambda x: x[0])
                    rows.append(current_row)
                current_row = [bubble]
            last_y = bubble[1]
            
        # Add last row if complete
        if len(current_row) == self.OPTIONS_PER_QUESTION:
            current_row.sort(key=lambda x: x[0])
            rows.append(current_row)
            
        return rows

    def detect_bubbles(self, preprocessed: np.ndarray, original_image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        # Create visualization image
        vis_image = original_image.copy()
        
        min_area = np.pi * (self.min_bubble_size / 2) ** 2 * 0.5
        max_area = np.pi * (self.max_bubble_size / 2) ** 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if (circularity > 0.7 and area > min_area and area < max_area):
                x, y, w, h = cv2.boundingRect(contour)
                bubbles.append((x, y, w, h))
        
        # Split bubbles into columns and validate structure
        columns = self.split_into_columns(bubbles)
        valid_bubbles = []
        
        for col_idx, column in enumerate(columns):
            rows = self.organize_column_bubbles(column)
            
            # Only keep bubbles that fit the expected grid structure
            for row_idx, row in enumerate(rows):
                if len(row) == self.OPTIONS_PER_QUESTION:
                    valid_bubbles.extend(row)
                    
                    # Calculate question number
                    question_num = col_idx * self.QUESTIONS_PER_COLUMN + row_idx + 1
                    
                    # Draw bubbles and process marks
                    for opt_idx, bubble in enumerate(row):
                        x, y, w, h = bubble
                        option = chr(65 + opt_idx)
                        
                        # Blue rectangle for all detected bubbles
                        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        # Check if bubble is marked
                        roi = preprocessed[y:y+h, x:x+w]
                        is_marked = np.mean(roi) > 127
                        
                        if is_marked:
                            # Red dot for marked answers
                            center = (x + w//2, y + h//2)
                            radius = min(w, h) // 4
                            cv2.circle(vis_image, center, radius, (0, 0, 255), -1)
                        
                        # Green outline for correct answers
                        if self.correct_answers.get(question_num) == option:
                            cv2.rectangle(vis_image, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
        
        return valid_bubbles, vis_image
    
    def analyze_answers(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Tuple[Dict[int, str], Dict[int, float]]:
        answers = {}
        confidence = {}
        
        # Split bubbles into columns
        columns = self.split_into_columns(bubbles)
        
        for col_idx, column in enumerate(columns):
            rows = self.organize_column_bubbles(column)
            
            for row_idx, row in enumerate(rows):
                if len(row) == self.OPTIONS_PER_QUESTION:
                    question_num = col_idx * self.QUESTIONS_PER_COLUMN + row_idx + 1
                    max_darkness = 0
                    selected_option = None
                    
                    for opt_idx, bubble in enumerate(row):
                        x, y, w, h = bubble
                        roi = image[y:y+h, x:x+w]
                        darkness = np.mean(roi)
                        
                        if darkness > max_darkness:
                            max_darkness = darkness
                            selected_option = chr(65 + opt_idx)
                    
                    if question_num <= self.TOTAL_QUESTIONS:
                        answers[question_num] = selected_option
                        confidence[question_num] = max_darkness / 255
        
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
