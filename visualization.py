import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List

def display_debug_pipeline(debug_images: Dict[str, np.ndarray]):
    """Display the debug pipeline with all processing steps"""
    st.write("Processing Steps:")
    
    for step_name, image in debug_images.items():
        with st.expander(f"Step: {step_name.replace('_', ' ').title()}"):
            st.image(image, use_column_width=True)

def plot_detection_results(result):
    """Create a visualization of the detected answers"""
    answers = result.answers
    confidence = result.confidence
    
    # Create figure
    fig = go.Figure()
    
    # Add answers plot
    questions = list(answers.keys())
    answer_values = list(answers.values())
    confidence_values = [confidence[q] * 100 for q in questions]
    
    # Add answers as markers
    fig.add_trace(go.Scatter(
        x=questions,
        y=answer_values,
        mode='markers+text',
        name='Selected Answer',
        text=answer_values,
        textposition="top center",
        marker=dict(
            size=20,
            color=confidence_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence %")
        )
    ))
    
    # Update layout
    fig.update_layout(
        title="Detected Answers with Confidence Levels",
        xaxis_title="Question Number",
        yaxis_title="Selected Option",
        yaxis=dict(
            tickmode='array',
            ticktext=['A', 'B', 'C', 'D', 'E'],
            tickvals=['A', 'B', 'C', 'D', 'E']
        ),
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
