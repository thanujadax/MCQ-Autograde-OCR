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
    TOTAL_QUESTIONS = 90
    OPTIONS_PER_QUESTION = 5
    TOTAL_BUBBLES = TOTAL_QUESTIONS * OPTIONS_PER_QUESTION

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

    def detect_bubbles(self, preprocessed: np.ndarray, original_image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bubbles = []
        vis_image = original_image.copy()
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_bubble_size * self.min_bubble_size or area > self.max_bubble_size * self.max_bubble_size:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.3:  # More lenient circularity threshold
                x, y, w, h = cv2.boundingRect(contour)
                bubbles.append((x, y, w, h))
        
        # Sort bubbles by position (top-to-bottom, left-to-right)
        sorted_bubbles = sorted(bubbles, key=lambda b: (b[1] // (self.max_bubble_size * 2), b[0]))
        
        # Keep only the first TOTAL_BUBBLES (450) bubbles
        valid_bubbles = sorted_bubbles[:self.TOTAL_BUBBLES]
        
        # Visualize bubbles
        for x, y, w, h in valid_bubbles:
            # Blue rectangle for all detected bubbles
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Check if bubble is marked
            roi = preprocessed[y:y+h, x:x+w]
            is_marked = np.mean(roi) > 127
            
            if is_marked:
                # Red dot for marked answers
                center = (x + w//2, y + h//2)
                radius = min(w, h) // 3  # Slightly larger red dot
                cv2.circle(vis_image, center, radius, (0, 0, 255), -1)
            
            # Calculate question number and option
            bubble_index = valid_bubbles.index((x, y, w, h))
            question_num = (bubble_index // self.OPTIONS_PER_QUESTION) + 1
            option = chr(65 + (bubble_index % self.OPTIONS_PER_QUESTION))
            
            # Green outline for correct answers
            if self.correct_answers.get(question_num) == option:
                cv2.rectangle(vis_image, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 3)  # Thicker green outline
        
        return valid_bubbles, vis_image

    def analyze_answers(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Tuple[Dict[int, str], Dict[int, float]]:
        answers = {}
        confidence = {}
        
        # Process bubbles in groups of 5 (one group per question)
        for q in range(self.TOTAL_QUESTIONS):
            question_bubbles = bubbles[q * self.OPTIONS_PER_QUESTION : (q + 1) * self.OPTIONS_PER_QUESTION]
            
            if len(question_bubbles) == self.OPTIONS_PER_QUESTION:
                max_darkness = 0
                selected_option = None
                
                for opt_idx, (x, y, w, h) in enumerate(question_bubbles):
                    roi = image[y:y+h, x:x+w]
                    darkness = np.mean(roi)
                    
                    if darkness > max_darkness:
                        max_darkness = darkness
                        selected_option = chr(65 + opt_idx)
                
                question_num = q + 1
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