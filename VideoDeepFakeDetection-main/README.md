# VideoDeepFakeDetection

Application that detects the originality of video files with artificial intelligence.

## Setup Environment

```bash
# Make sure your PIP is up to date
pip install -U pip wheel setuptools

# Install required dependencies
pip install -r requirements.txt
```

## Application

- You can run the file named [main.py](main.py).
- Running on http://127.0.0.1:5000

## Overview

1- The video file is opened, and various video properties such as fps, width, and height are obtained.

2- Face detection is performed using **MTCNN (Multi-Task Cascaded Convolutional Networks)**.

3- The detected face is transformed into a feature vector using a pre-trained **Inception Resnet V1 model (InceptionResnetV1)**.

4- A comparison is made with the face in the previous frame, and a similarity score is calculated.

5- Similarity scores below a certain threshold are considered as indicative of a deepfake.

6- If deepfakes are detected in a consecutive number of frames, it is marked as a deepfake, and a frame is added to the video.

7- Processed frames are written to an output video file.

