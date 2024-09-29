import base64
import streamlit as st
import numpy as np
from collections import deque
import cv2
sequence = deque(maxlen=128)  
Q = deque(maxlen=128)

def set_background(image_file):
  
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def detect_violence(frame,model):
    print("Loading model ...")

    frame_resized = cv2.resize(frame, (64, 64)).astype("float32") / 255
    sequence.append(frame_resized)
    print(f"the size of sequesse : {len(sequence)}")

    if len(sequence) == 16:  # Check if we have collected 16 frames
        print("frame has been collected")
        input_sequence = np.expand_dims(np.array(sequence), axis=0)  # Shape (1, 16, 64, 64, 3)
        preds = model.predict(input_sequence)[0]
        Q.append(preds)
        sequence.popleft()  # Remove the oldest frame to maintain a sliding window of 16 frames
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = "Violence" if i == 1 else "No Violence"
        confidence = results[i] * 100
        return label, confidence

    return "No Violence", 0.0 