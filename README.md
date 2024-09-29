# Real-Time Violence Detection Using Streamlit

## Overview

This project implements a **real-time violence detection** system using machine learning and deep learning techniques. The application is built with **Streamlit** and leverages a pre-trained **LSTM model** to classify violence in uploaded videos. The system processes video files, analyzes frames, and provides immediate feedback on whether violence is detected, along with confidence levels.

The application is designed for scenarios where rapid detection of violence in videos is required, such as in security, surveillance, or content moderation.

## Key Features

- **Real-Time Video Processing**: Processes video files frame-by-frame and analyzes the content for signs of violence.
- **Machine Learning Integration**: Uses an LSTM model for violence detection, providing accurate classification.
- **Interactive UI**: Built with Streamlit, offering an intuitive and easy-to-use interface for uploading videos and viewing results.
- **Live Feedback**: Displays the video while processing, with overlays showing detection results and confidence scores in real-time.
- **Customizable Settings**: Adjust the confidence threshold for violence detection through a user-friendly sidebar.
- **Creative Design**: Adaptive background and sleek UI design for a modern, professional appearance.

## Technologies Used

- **Python**: Core programming language for building the app.
- **Streamlit**: For creating the interactive web interface.
- **Keras**: Used for loading the pre-trained LSTM model.
- **OpenCV**: For real-time video processing and frame analysis.
- **Streamlit-WebRTC**: For enabling real-time video streaming (if integrated).
- **Numpy**: For handling frame sequences and model predictions.
- **AI Model**: A pre-trained LSTM model designed for violence classification.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/violence-detection.git
   cd violence-detection
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Additional packages**: 
   If you are using WebRTC for real-time streaming, install additional dependencies:
   ```bash
   pip install streamlit-webrtc aiortc opencv-python-headless
   ```

## Usage

1. Upload a video file in the format (`mp4`, `avi`, or `mov`).
2. Adjust the **confidence threshold** in the sidebar to control sensitivity.
3. Watch the video playback with real-time violence classification. The result is overlaid on the video with confidence percentages.
4. View detailed logs and statistics for each frame in the sidebar (optional).

## File Structure

- **app.py**: Main Streamlit app for running the interface and video processing.
- **util.py**: Contains helper functions such as `detect_violence` and `set_background`.
- **weights/**: Directory containing the pre-trained LSTM model weights.
- **photo/**: Directory for storing background images used in the app.

## Future Enhancements

- **WebRTC integration**: Add live video feed support for real-time camera analysis.
- **Improved Model**: Train the model further on more diverse datasets for better accuracy.
- **Notification System**: Add alerts or email notifications when violence is detected.

## Contribution

1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request with detailed descriptions of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to open-source communities for providing the tools and frameworks used in this project.
- Thanks to [Streamlit](https://streamlit.io/) for making web app development intuitive and easy.
