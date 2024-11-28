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
    
    # Processing parameters
    threshold = st.sidebar.slider("Threshold", 0, 255, 128)
    bubble_size = st.sidebar.slider("Bubble Size", 10, 50, 20)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    
    # File upload
    uploaded_file = st.file_uploader("Choose an answer sheet image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Create processor instance
        processor = MCQProcessor(
            threshold=threshold,
            bubble_size=bubble_size,
            contrast=contrast
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
