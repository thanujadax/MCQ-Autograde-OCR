import streamlit as st
import cv2
import numpy as np
from processing import MCQProcessor
from visualization import (
    display_debug_pipeline,
    plot_detection_results
)
from utils import load_image, export_to_csv
import tempfile
import os

st.set_page_config(layout="wide", page_title="MCQ Answer Sheet Processor")

def main():
    st.title("MCQ Answer Sheet Processor")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    
    # Processing parameters with tooltips
    threshold = st.sidebar.slider(
        "Darkness Threshold",
        0, 255, 128,
        help="Controls how dark a marked bubble needs to be to be considered filled"
    )
    
    min_bubble_size = st.sidebar.slider(
        "Minimum Bubble Size",
        0, 50, 20,
        help="Minimum size (in pixels) of bubbles to detect"
    )
    
    max_bubble_size = st.sidebar.slider(
        "Maximum Bubble Size",
        0, 75, 35,
        help="Maximum size (in pixels) of bubbles to detect"
    )
    
    contrast = st.sidebar.slider(
        "Contrast Enhancement",
        0.5, 2.0, 1.0,
        help="Adjust image contrast to improve bubble detection"
    )
    
    # Correct answers input
    correct_answers_input = st.sidebar.text_input(
        "Correct Answers",
        help="Enter correct answers in format '1:A,2:B,3:C'"
    )
    
    # Parse correct answers
    correct_answers = {}
    if correct_answers_input:
        try:
            for answer in correct_answers_input.split(','):
                q, a = answer.strip().split(':')
                correct_answers[int(q)] = a.strip().upper()
        except ValueError:
            st.sidebar.warning("Please use the format '1:A,2:B,3:C' for correct answers")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an answer sheet image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Create processor instance
        processor = MCQProcessor(
            threshold=threshold,
            min_bubble_size=min_bubble_size,
            max_bubble_size=max_bubble_size,
            contrast=contrast,
            correct_answers=correct_answers
        )
        
        # Load and process image
        image = load_image(uploaded_file)
        
        # Process image
        with st.spinner("Processing image..."):
            try:
                result, debug_images = processor.process(image)
                
                # Display processed image with overlaid detections
                st.subheader("Detected Answer Sheet")
                st.image(debug_images['detected_bubbles'], use_container_width=True)
                
                # Display answer results
                st.subheader("Detected Answers")
                plot_detection_results(result)
                
                # Debug pipeline visualization
                st.subheader("Processing Pipeline")
                display_debug_pipeline(debug_images)
                
                # Export results
                if st.button("Export Results"):
                    csv_path = export_to_csv(result)
                    with open(csv_path, 'r') as file:
                        st.download_button(
                            label="Download CSV",
                            data=file,
                            file_name="mcq_results.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
