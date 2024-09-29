import streamlit as st
from keras.models import load_model
import cv2
import tempfile
from util import detect_violence, set_background
from streamlit_webrtc import webrtc_streamer

# Set page configuration for a more professional look
st.set_page_config(page_title="Real-Time Video Analysis", layout="wide", initial_sidebar_state="expanded")

# Set a custom background image
set_background('photo/v.jpg')

# Sidebar for user inputs
with st.sidebar:
    st.title("Settings")
    st.write("Upload video for real-time analysis.")
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    st.write("Classification Options:")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Main title and description
st.markdown("<h1 style='text-align: center; color: white;'>Real-Time Violence Classification</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #f1c40f;'>Analyze video streams for potential violence</h3>", unsafe_allow_html=True)

# Load the classifier model
model = load_model('weight/my_model_lstm.h5')

# Placeholder for real-time video display
stframe = st.empty()  # This will hold the video frames
progress_bar = st.progress(0)  # To show the progress of processing

# Process the uploaded video
if video_file is not None:
    temp_video_file = tempfile.NamedTemporaryFile(delete=False)
    temp_video_file.write(video_file.read())

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(temp_video_file.name)

    # Get total frames for the progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame processing loop
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform violence detection on the frame
        label, conf = detect_violence(frame, model)
        
        # Draw label and confidence on the frame
        if conf >= confidence_threshold:
            text = f"Violence Detected: {label} ({conf:.2f}%)"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update the frame in the placeholder
        stframe.image(frame, channels="RGB", use_column_width=True)

        # Update the progress bar
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)
    
    cap.release()
else:
    st.info("Please upload a video file to start the real-time classification.")