#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("The following cell has the emotion detection algorithm")
# UNIT 1, Working perfectly


# In[2]:


get_ipython().system('pip install cv2')

import cv2
import dlib
import numpy as np
from IPython.display import display, clear_output
from scipy.spatial import distance
from imutils import face_utils
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import csv

# Function to detect eyes in a frame
def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if faces:
        shape = predictor(gray, faces[0])
        left_eye = shape.parts()[36:42]
        right_eye = shape.parts()[42:48]
        return left_eye, right_eye
    else:
        return None, None

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    eye = np.array([(point.x, point.y) for point in eye])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/scripts/shape_predictor_68_face_landmarks.dat")

# Load emotion detection model
emotion_model = load_model('/scripts/video.h5')

emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize video capture from the camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it if you have multiple cameras)

# Get video properties for the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('/results/outputvideo.avi', fourcc, fps, (width, height))

# Initialize variables to record durations
duration_eyes_closed = 0
duration_looking_left = 0
duration_looking_right = 0
duration_looking_straight = 0

# Initialize variables for counting eye movement
count_left = 0
count_right = 0
count_straight = 0

# Load face detector and shape predictor for emotion detection
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("/scripts/face_landmarks.dat")

# Initialize head pose estimation
official_start_time = time.time()
start_time = time.time()
end_time = 0

#Variables to track emotion detected
emotion_start_time = time.time()
e_start_time = time.time()
e_end_time = 0
angry_emotion = 0
sad_emotion = 0
happy_emotion = 0
fear_emotion = 0
disgust_emotion = 0
neutral_emotion = 0
surprise_emotion = 0

# Variables to track time spent in different head pose directions
time_forward_seconds = 0
time_left_seconds = 0
time_right_seconds = 0
time_up_seconds = 0
time_down_seconds = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Eye tracking
    left_eye, right_eye = detect_eyes(frame)

    if left_eye is not None and right_eye is not None:
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)

        # Calculate the average EAR for both eyes
        avg_ear = (ear_left + ear_right) / 2.0

        # Set a threshold for distraction detection (you may need to adjust this)
        distraction_threshold = 0.2

        # Check if the person is distracted
        if avg_ear < distraction_threshold:
            cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            duration_eyes_closed += 1 / fps  # Increment the duration
            count_straight += 1

        else:
            # Check gaze direction
            horizontal_ratio = (left_eye[0].x + right_eye[3].x) / 2 / width
            if horizontal_ratio < 0.4:
                cv2.putText(frame, "Looking Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                duration_looking_left += 1 / fps  # Increment the duration
                count_left += 1
            elif horizontal_ratio > 0.6:
                cv2.putText(frame, "Looking Right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                duration_looking_right += 1 / fps  # Increment the duration
                count_right += 1
            else:
                cv2.putText(frame, "Looking Straight", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                duration_looking_straight += 1 / fps  # Increment the duration

        # Draw contours around eyes
        for eye in [left_eye, right_eye]:
            for point in eye:
                x, y = point.x, point.y
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = shape_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y + h, x:x + w]
        face = zoom(face, (48 / face.shape[0], 48 / face.shape[1]))
        face = face.astype(np.float32)
        face /= float(face.max())
        face = np.reshape(face.flatten(), (1, 48, 48, 1))

        prediction = emotion_model.predict(face)
        prediction_result = np.argmax(prediction)

        # Rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Annotate main image with emotion label
        if prediction_result == 0:
            cv2.putText(frame, "Angry", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            angry_emotion += time.time() - e_start_time
            e_start_time = time.time()
        elif prediction_result == 1:
            cv2.putText(frame, "Disgust", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            disgust_emotion += time.time() - e_start_time
            e_start_time = time.time()
        elif prediction_result == 2:
            cv2.putText(frame, "Fear", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            fear_emotion += time.time() - e_start_time 
            e_start_time = time.time()
        elif prediction_result == 3:
            cv2.putText(frame, "Happy", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            happy_emotion += time.time() - e_start_time 
            e_start_time = time.time()
        elif prediction_result == 4:
            cv2.putText(frame, "Sad", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            sad_emotion += time.time() - e_start_time 
            e_start_time = time.time()
        elif prediction_result == 5:
            cv2.putText(frame, "Surprise", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            surprise_emotion += time.time() - e_start_time 
            e_start_time = time.time()
        else:
            cv2.putText(frame, "Neutral", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            neutral_emotion += time.time() - e_start_time 
            e_start_time = time.time()

    # Head pose estimation
    startTime = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #it was 1
#     frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            print(f"X Rotation: {angles[0] * 10000}")
            print(f"Y Rotation: {angles[1] * 10000}")

            if angles[1] * 10000 < -100:
                text = "Looking Left"
                time_left_seconds += time.time() - start_time
                start_time = time.time()

            elif angles[1] * 10000 > 100:
                text = "Looking Right"
                time_right_seconds += time.time() - start_time
                start_time = time.time()

            elif angles[0] * 10000 < -100:
                text = "Looking Down"
                time_down_seconds += time.time() - start_time
                start_time = time.time()

            elif angles[0] * 10000 > 200:
                text = "Looking Up"
                time_up_seconds += time.time() - start_time
                start_time = time.time()

            else:
                text = "Forward"
                time_forward_seconds += time.time() - start_time
                start_time = time.time()

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            cv2.putText(frame, text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Open the CSV file in write mode and append the angles to it
    with open('headPoses.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header row if the file is empty
        if file.tell() == 0:
            writer.writerow(["X Rotation", "Y Rotation"])

        # Write the angles to the CSV file
        #writer.writerow([angles[0] * 10000, angles[1] * 10000]) #bonbon

    output_video.write(frame)  # Write the frame to the output video

    # Display the frame without modifying color
    cv2.imshow('Frame', frame)
    # Clear the previous output
    clear_output(wait=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, video writer, and close all windows
cap.release()
output_video.release()
cv2.destroyAllWindows()


# Print the durations and most observed features for emotion detection
print(f"\nEmotion Detection:")
print(f"Duration of Happiness: {happy_emotion} seconds")
print(f"Duration of Sadness: {sad_emotion} seconds")
print(f"Duration of Disgust: {disgust_emotion} seconds")
print(f"Duration of Fear: {fear_emotion} seconds")
print(f"Duration of Anger: {angry_emotion} seconds")
print(f"Duration of Neutral: {neutral_emotion} seconds")
print(f"Duration of Surprise: {surprise_emotion} seconds")

# Determine the most observed emotions movement
max_eye_duration = max(happy_emotion, sad_emotion, disgust_emotion, fear_emotion, angry_emotion, neutral_emotion, surprise_emotion)
if max_eye_duration == happy_emotion:
    print("The most observed emotion: Happiness")
elif max_eye_duration == sad_emotion:
    print("The most observed emotion: Sadness")
elif max_eye_duration == disgust_emotion:
    print("The most observed emotion: Disgust")
elif max_eye_duration == fear_emotion:
    print("The most observed emotion: Fear")
elif max_eye_duration == angry_emotion:
    print("The most observed emotion: Anger")
elif max_eye_duration == surprise_emotion:
    print("The most observed emotion: Surprise")
else:
    print("The most observed emotion: Neutral")


# Print the durations and most observed features for eyes
print(f"\nEye Movements:")
print(f"Duration taken looking right: {duration_looking_right} sec")
print(f"Duration taken closed eyes: {duration_eyes_closed} sec")
print(f"Duration taken looking left: {duration_looking_left} sec")
print(f"Duration taken looking straight: {duration_looking_straight} sec")

# Determine the most observed eye movement
max_eye_duration = max(duration_looking_right, duration_eyes_closed, duration_looking_left, duration_looking_straight)
if max_eye_duration == duration_looking_right:
    print("The most observed eye movement: Looking Right")
elif max_eye_duration == duration_eyes_closed:
    print("The most observed eye movement: Eyes Closed")
elif max_eye_duration == duration_looking_left:
    print("The most observed eye movement: Looking Left")
else:
    print("The most observed eye movement: Looking Straight")

# Print the durations and most observed features for head pose
print(f"\nHead Pose Estimation:")
print(f"Duration of Time Looking Forward: {time_forward_seconds} seconds")
print(f"Duration of Time Looking Up: {time_up_seconds} seconds")
print(f"Duration of Time Looking Left: q{time_left_seconds} seconds")
print(f"Duration of Time Looking Right: {time_right_seconds} seconds")
print(f"Duration of Time Looking Down: {time_down_seconds} seconds")

# Determine the most observed eye movement
max_eye_duration = max(time_forward_seconds, time_up_seconds, time_left_seconds, time_right_seconds, time_down_seconds)
if max_eye_duration == time_forward_seconds:
    print('\033[93m'+"The most observed head pose: Facing Forward")
elif max_eye_duration == time_up_seconds:
    print('\033[93m'+"The most observed head pose: Facing Upwards")
elif max_eye_duration == time_left_seconds:
    print('\033[93m'+"The most observed head pose: Facing Left")
elif max_eye_duration == time_right_seconds:
    print('\033[93m'+"The most observed head pose: Facing Right")
else:
    print('\033[93m'+"The most observed head pose: Facing Downwards")


# In[ ]:


print("The following cell has multiple face detection algorithm")
# UNIT 2, Working perfectly


# In[ ]:


import cv2
import dlib

# Initialize the video capture (use 0 for the first camera device)
cap = cv2.VideoCapture(0)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

person_count = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each detected face
    for i, face in enumerate(faces):
        # Draw a rectangle around the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Label the face with a unique identifier
        label = f'Person {i+1}'
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:


# Test code 1
print("Simple code to read the .h5 file")


# In[ ]:


import h5py

# Path to the HDF5 file
h5_file_path = 'video_modified.h5'

# Open the HDF5 file for reading
with h5py.File(h5_file_path, 'r') as f:
    print("Keys in the HDF5 file:")
    print("======================")
    # Print all groups and datasets
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    f.visititems(print_attrs)


# In[ ]:


# FINAL, Test success
print("The following cell has the integrated code which is final one of version 1")


# In[1]:


get_ipython().system('pip3 install opencv-python-headless')
get_ipython().system('pip3 install dlib')
get_ipython().system('pip3 install numpy')
get_ipython().system('pip3 install ipython')
get_ipython().system('pip3 install scipy')
get_ipython().system('pip3 install imutils')
get_ipython().system('pip3 install tensorflow')
get_ipython().system('pip3 install mediapipe')

import os
import cv2
import dlib
import numpy as np
from IPython.display import display, clear_output
from scipy.spatial import distance
from imutils import face_utils
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import csv
import tensorflow as tf

# Custom deserialization function for VarianceScaling
def custom_VarianceScaling_deserializer(config):
    from tensorflow.keras.initializers import VarianceScaling
    # Remove 'dtype' from config if it exists
    config.pop('dtype', None)
    return VarianceScaling(**config)

# Register the custom deserializer
tf.keras.utils.get_custom_objects().update({'VarianceScaling': custom_VarianceScaling_deserializer})

# Function to detect eyes in a frame
def detect_eyes(frame, shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    return left_eye, right_eye

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    eye = np.array([(point[0], point[1]) for point in eye])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to get the specified file's path
def get_abs_path(directory, file):
    directory_path = os.path.join(os.getcwd(), '..', directory)
    file_path = os.path.join(directory_path, file)
    return file_path

# Load dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(get_abs_path('scripts', 'shape_predictor_68_face_landmarks.dat'))

# Load emotion detection model
emotion_model = load_model(get_abs_path('scripts', 'FER_model.h5'))

emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize video capture from the camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it if you have multiple cameras)

# Get video properties for the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(get_abs_path('results', 'outputvideo.avi'), fourcc, fps, (width, height))

# Initialize variables to record durations
duration_eyes_closed = {}
duration_looking_left = {}
duration_looking_right = {}
duration_looking_straight = {}

# Initialize variables for counting eye movement
count_left = {}
count_right = {}
count_straight = {}

# Variables to track emotion detected
emotion_start_time = time.time()
emotion_duration = {"angry": {}, "sad": {}, "happy": {}, "fear": {}, "disgust": {}, "neutral": {}, "surprise": {}}

# Variables to track time spent in different head pose directions
time_forward_seconds = {}
time_left_seconds = {}
time_right_seconds = {}
time_up_seconds = {}
time_down_seconds = {}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        person_id = f"Person {i+1}"

        if person_id not in duration_eyes_closed:
            duration_eyes_closed[person_id] = 0
            duration_looking_left[person_id] = 0
            duration_looking_right[person_id] = 0
            duration_looking_straight[person_id] = 0
            count_left[person_id] = 0
            count_right[person_id] = 0
            count_straight[person_id] = 0
            time_forward_seconds[person_id] = 0
            time_left_seconds[person_id] = 0
            time_right_seconds[person_id] = 0
            time_up_seconds[person_id] = 0
            time_down_seconds[person_id] = 0
            for emotion in emotion_duration:
                emotion_duration[emotion][person_id] = 0

        # Eye tracking
        left_eye, right_eye = detect_eyes(frame, shape)

        if left_eye is not None and right_eye is not None:
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)

            # Calculate the average EAR for both eyes
            avg_ear = (ear_left + ear_right) / 2.0

            # Set a threshold for distraction detection (you may need to adjust this)
            distraction_threshold = 0.2

            # Check if the person is distracted
            if avg_ear < distraction_threshold:
                cv2.putText(frame, f"{person_id}: Eyes Closed", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                duration_eyes_closed[person_id] += 1 / fps  # Increment the duration
                count_straight[person_id] += 1

            else:
                # Check gaze direction
                horizontal_ratio = (left_eye[0][0] + right_eye[3][0]) / 2 / width
                if horizontal_ratio < 0.4:
                    cv2.putText(frame, f"{person_id}: Looking Left", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    duration_looking_left[person_id] += 1 / fps  # Increment the duration
                    count_left[person_id] += 1
                elif horizontal_ratio > 0.6:
                    cv2.putText(frame, f"{person_id}: Looking Right", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    duration_looking_right[person_id] += 1 / fps  # Increment the duration
                    count_right[person_id] += 1
                else:
                    cv2.putText(frame, f"{person_id}: Looking Straight", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    duration_looking_straight[person_id] += 1 / fps  # Increment the duration

            # Draw contours around eyes
            for eye in [left_eye, right_eye]:
                for point in eye:
                    x, y = point[0], point[1]
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Emotion detection
        (x, y, w, h) = face_utils.rect_to_bb(face)
        face_crop = gray[y:y + h, x:x + w]
        face_crop = zoom(face_crop, (48 / face_crop.shape[0], 48 / face_crop.shape[1]))
        face_crop = face_crop.astype(np.float32)
        face_crop /= float(face_crop.max())
        face_crop = np.reshape(face_crop.flatten(), (1, 48, 48, 1))

        prediction = emotion_model.predict(face_crop)
        prediction_result = np.argmax(prediction)

        # Rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Annotate main image with emotion label
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotion_label = emotion_labels[prediction_result]
        cv2.putText(frame, f"{person_id}: {emotion_label}", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        emotion_duration[emotion_label.lower()][person_id] += time.time() - emotion_start_time
        emotion_start_time = time.time()

        # Head pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
              for idx, lm in enumerate(face_landmarks.landmark):
                  if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                      if idx == 1:
                          nose_2d = (lm.x * img_w, lm.y * img_h)
                          nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                      x, y = int(lm.x * img_w), int(lm.y * img_h)

                      # Get the 2D Coordinates
                      face_2d.append([x, y])

                      # Get the 3D Coordinates
                      face_3d.append([x, y, lm.z])

          face_2d = np.array(face_2d, dtype=np.float64)
          face_3d = np.array(face_3d, dtype=np.float64)

          # Camera matrix
          focal_length = 1 * img_w
          cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                [0, focal_length, img_h / 2],
                                [0, 0, 1]])

          # Distortion parameters
          dist_matrix = np.zeros((4, 1), dtype=np.float64)

          # Solve PnP
          success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

          # Get rotational matrix
          rmat, jac = cv2.Rodrigues(rot_vec)

          # Get angles
          angles, mtx_r, mtx_q, qx, qy, qz = cv2.RQDecomp3x3(rmat)

          # Get the y rotation degree
          x_angle = angles[0] * 360
          y_angle = angles[1] * 360
          z_angle = angles[2] * 360

          # See where the user's head tilting
          if y_angle < -10:
              text = "Looking Left"
              time_left_seconds[person_id] += 1 / fps
          elif y_angle > 10:
              text = "Looking Right"
              time_right_seconds[person_id] += 1 / fps
          elif x_angle < -10:
              text = "Looking Down"
              time_down_seconds[person_id] += 1 / fps
          elif x_angle > 10:
              text = "Looking Up"
              time_up_seconds[person_id] += 1 / fps
          else:
              text = "Looking Forward"
              time_forward_seconds[person_id] += 1 / fps

          # Display the text
          cv2.putText(frame, f"{person_id}: {text}", (500, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame in a window
    cv2.imshow('Frame', frame)

    # Clear the output in Jupyter notebook
    clear_output(wait=True)
    display(frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

with open(get_abs_path('results', 'eye_tracking_data.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Person ID", "Duration Eyes Closed (s)", "Duration Looking Left (s)", "Duration Looking Right (s)", "Duration Looking Straight (s)", "Left Counts", "Right Counts", "Straight Counts"])
    for person_id in duration_eyes_closed:
        writer.writerow([person_id, duration_eyes_closed[person_id], duration_looking_left[person_id], duration_looking_right[person_id], duration_looking_straight[person_id], count_left[person_id], count_right[person_id], count_straight[person_id]])

with open(get_abs_path('results', 'emotion_detection_data.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Person ID", "Angry (s)", "Sad (s)", "Happy (s)", "Fear (s)", "Disgust (s)", "Neutral (s)", "Surprise (s)"])
    for person_id in emotion_duration["angry"]:
        writer.writerow([person_id, emotion_duration["angry"][person_id], emotion_duration["sad"][person_id], emotion_duration["happy"][person_id], emotion_duration["fear"][person_id], emotion_duration["disgust"][person_id], emotion_duration["neutral"][person_id], emotion_duration["surprise"][person_id]])

with open(get_abs_path('results', 'head_pose_data.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Person ID", "Looking Forward (s)", "Looking Left (s)", "Looking Right (s)", "Looking Up (s)", "Looking Down (s)"])
    for person_id in time_forward_seconds:
        writer.writerow([person_id, time_forward_seconds[person_id], time_left_seconds[person_id], time_right_seconds[person_id], time_up_seconds[person_id], time_down_seconds[person_id]])


# In[ ]:




