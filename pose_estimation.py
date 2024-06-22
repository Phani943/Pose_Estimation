import cv2
import time
import tempfile
import numpy as np
import mediapipe as mp
import streamlit as st
import tensorflow as tf

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32)
]


@st.cache_resource
def load_model():
    return tf.saved_model.load("Models/ssd_mobilenet/saved_model")


model = load_model()
mp_pose = mp.solutions.pose

labels = {1: 'person'}


def detect_persons(image):
    tensor_img = tf.convert_to_tensor(image)
    tensor_img = tensor_img[tf.newaxis, ...]

    detections = model(tensor_img)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    return boxes, scores, classes


def draw_landmarks(img, landmarks):
    height, width, _ = img.shape
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if landmarks.landmark[start_idx] and landmarks.landmark[end_idx]:
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]

            start_coordinates = (int(start_point.x * width), int(start_point.y * height))
            end_coordinates = (int(end_point.x * width), int(end_point.y * height))

            cv2.line(img, start_coordinates, end_coordinates, (0, 255, 0), 2)

    return img


def draw_bounding_box(img, box, width, height):
    y_min, x_min, y_max, x_max = box
    left, right, top, bottom = x_min * width, x_max * width, y_min * height, y_max * height
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)


def process_frame(frame, pose, draw_box):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores, classes = detect_persons(image_rgb)

    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:
            y_min, x_min, y_max, x_max = boxes[i]
            left, right, top, bottom = x_min * width, x_max * width, y_min * height, y_max * height
            person_roi = frame[int(top):int(bottom), int(left):int(right)]

            results = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                person_roi = draw_landmarks(person_roi, results.pose_landmarks)

            frame[int(top):int(bottom), int(left):int(right)] = person_roi
            if draw_box:
                draw_bounding_box(frame, boxes[i], width, height)

    return frame


def main():
    st.markdown(
        """
        <style>
            .title {
                font-size: 36px;
                color: #000000;
                padding-bottom: 40px;
                border-bottom: 4px solid #000000;
            }
            .intro {
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .upload-section {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .button-primary {
                background-color: #008CBA;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                transition: background-color 0.3s ease;
                text-align: center;
                display: inline-block;
                cursor: pointer;
            }
            .button-primary:hover {
                background-color: #005f7f;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p class='title'>Multi-Person Pose Estimation</p>", unsafe_allow_html=True)
    st.markdown("<p class='intro'>Choose an operation type:</p>", unsafe_allow_html=True)

    operation_type = st.radio("Choose operation type", ("Input", "Demo"))

    pose = mp_pose.Pose()

    if operation_type == "Input":
        input_type = st.radio("Choose input type", ("Image", "Video"))

        if input_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload an image file (.jpg, .jpeg, .png)",
                type=["jpg", "jpeg", "png"]
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a video file (.mp4, .mov, .avi, .mkv)",
                type=["mp4", "mov", "avi", "mkv"]
            )

        draw_box = st.checkbox("Draw bounding box", value=False)

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name

            if input_type == "Video":
                cam = cv2.VideoCapture(file_path)
                st_frame = st.empty()

                while cam.isOpened():
                    success, frame = cam.read()
                    if not success:
                        break

                    frame = process_frame(frame, pose, draw_box)

                    st_frame.image(frame, channels='BGR', use_column_width=True)
                    time.sleep(1)
                    st.empty()

                st.text("Completed")
                cam.release()

            elif input_type == "Image":
                image = cv2.imread(file_path)
                processed_image = process_frame(image, pose, draw_box)

                st.image(processed_image, channels='BGR', use_column_width=True)

    elif operation_type == "Demo":
        st.empty()
        st.markdown("<p class='intro'>Demo video will be shown below:</p>", unsafe_allow_html=True)

        demo_image_path = "Images/demo.jpg"

        image = cv2.imread(demo_image_path)
        processed_image = process_frame(image, pose, draw_box=False)

        st.image(processed_image, channels='BGR', use_column_width=True)
        st.text("Done")


if __name__ == "__main__":
    main()
