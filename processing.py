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
            
            # Get bubble index and question info
            bubble_index = valid_bubbles.index((x, y, w, h))
            question_num = (bubble_index // self.OPTIONS_PER_QUESTION) + 1
            option = chr(65 + (bubble_index % self.OPTIONS_PER_QUESTION))
            
            # Check if bubble meets marking criteria
            roi = preprocessed[y:y+h, x:x+w]
            darkness = np.mean(roi)
            
            # Get darkness values for all options in this question
            question_start = (question_num - 1) * self.OPTIONS_PER_QUESTION
            question_end = question_start + self.OPTIONS_PER_QUESTION
            question_bubbles = valid_bubbles[question_start:question_end]
            darkness_values = []
            
            for qx, qy, qw, qh in question_bubbles:
                q_roi = preprocessed[qy:qy+qh, qx:qx+qw]
                darkness_values.append(np.mean(q_roi))
            
            # Check if this bubble is significantly darker
            max_darkness = max(darkness_values)
            second_max = sorted(darkness_values)[-2] if len(darkness_values) > 1 else 0
            is_marked = (darkness > 127 and 
                        darkness == max_darkness and 
                        (max_darkness - second_max) > 30)
            
            if is_marked:
                # Red rectangle for marked answers meeting criteria
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Blue filled rectangle for correct answers
            if self.correct_answers.get(question_num) == option:
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), -1)
        
        return valid_bubbles, vis_image

    def analyze_answers(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Tuple[Dict[int, str], Dict[int, float]]:
        answers = {}
        confidence = {}
        
        # Thresholds for answer detection
        darkness_threshold = 127  # Minimum darkness to consider marked
        min_darkness_diff = 30   # Minimum difference from other options
        
        # Process bubbles in groups of 5 (one group per question)
        for q in range(self.TOTAL_QUESTIONS):
            question_bubbles = bubbles[q * self.OPTIONS_PER_QUESTION : (q + 1) * self.OPTIONS_PER_QUESTION]
            darkness_values = []
            
            if len(question_bubbles) == self.OPTIONS_PER_QUESTION:
                # Calculate darkness for each option
                for x, y, w, h in question_bubbles:
                    roi = image[y:y+h, x:x+w]
                    darkness = np.mean(roi)
                    darkness_values.append(darkness)
                
                # Find the darkest bubble and second darkest
                max_darkness = max(darkness_values)
                second_max = sorted(darkness_values)[-2] if len(darkness_values) > 1 else 0
                
                # Check if the darkest bubble meets our criteria
                if max_darkness > darkness_threshold and (max_darkness - second_max) > min_darkness_diff:
                    selected_idx = darkness_values.index(max_darkness)
                    question_num = q + 1
                    answers[question_num] = chr(65 + selected_idx)
                    confidence[question_num] = (max_darkness - second_max) / 255
                else:
                    # Mark as unanswered if criteria not met
                    question_num = q + 1
                    answers[question_num] = None
                    confidence[question_num] = 0.0
        
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
        
        # Create detected answers visualization
        answers_visualization = image.copy()
        for x, y, w, h in bubbles:
            bubble_index = bubbles.index((x, y, w, h))
            question_num = (bubble_index // self.OPTIONS_PER_QUESTION) + 1
            option = chr(65 + (bubble_index % self.OPTIONS_PER_QUESTION))
            
            roi = preprocessed[y:y+h, x:x+w]
            is_marked = np.mean(roi) > 127
            
            if is_marked or self.correct_answers.get(question_num) == option:
                if is_marked:
                    cv2.rectangle(answers_visualization, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if self.correct_answers.get(question_num) == option:
                    cv2.rectangle(answers_visualization, (x, y), (x+w, y+h), (255, 0, 0), -1)
        
        debug_images['detected_answers'] = answers_visualization
        
        # Analyze answers
        answers, confidence = self.analyze_answers(preprocessed, bubbles)
        
        return ProcessingResult(answers, confidence, bubbles), debug_images