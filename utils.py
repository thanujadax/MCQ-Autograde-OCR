import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import tempfile
import os

def load_image(uploaded_file):
    """Load an uploaded image file and convert to numpy array"""
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Convert to RGB if RGBA
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image

def export_to_csv(result) -> str:
    """Export results to CSV file"""
    # Create DataFrame
    data = {
        'Question': list(result.answers.keys()),
        'Answer': list(result.answers.values()),
        'Confidence': [f"{conf * 100:.2f}%" for conf in result.confidence.values()]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, 'mcq_results.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path
