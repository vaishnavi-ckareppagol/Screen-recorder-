import cv2
import numpy as np
import pyautogui
from datetime import datetime

# Load pre-trained models
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Screen size and video writer setup
screen_size = pyautogui.size()
filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(filename, fourcc, 10.0, screen_size)

# Start webcam
cam = cv2.VideoCapture(0)

print("Recording... Press Ctrl+C or Stop button in Thonny to stop.")

try:
    while True:
        # Capture screen
        screen = pyautogui.screenshot()
        screen_frame = np.array(screen)
        screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2RGB)

        # Capture webcam frame
        ret, face_frame = cam.read()
        if ret:
            gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = face_frame[y:y+h, x:x+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                             (78.426, 87.768, 114.895), swapRB=False)

                gender_net.setInput(blob)
                gender = GENDER_LIST[gender_net.forward()[0].argmax()]

                age_net.setInput(blob)
                age = AGE_LIST[age_net.forward()[0].argmax()]

                label = f"{gender}, {age}"
                cv2.putText(screen_frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 2)

        # Add date/time overlay
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(screen_frame, timestamp, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 180, 0), 2)

        # Add watermark
        cv2.putText(screen_frame, "Vaishnavi's AI Recorder",
                    (screen_size[0]-450, screen_size[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

        # Save video frame
        out.write(screen_frame)

except KeyboardInterrupt:
    print(f"\nRecording stopped and saved as {filename}")
    out.release()
    cam.release()
    cv2.destroyAllWindows()