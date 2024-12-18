import cv2
import numpy as np
import mediapipe as mp
import os
import time

from ultralytics import YOLO


LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LANDMARKS = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]

CAMERA_DEVICE_NUMBER = 1

DETECTION_MODELS_ROOT_PATH = "d_models"
PREDICTION_MODELS_ROOT_PATH = "models"

MODEL_TYPE = "eyes"

DETECTION_MODEL_FILENAME = "face_landmarker.task"
PREDICTION_MODEL_FILENAME = "model-20241213-225015.pt"

DETECTION_MODEL_PATH = os.path.join(DETECTION_MODELS_ROOT_PATH, MODEL_TYPE, DETECTION_MODEL_FILENAME)
PREDICTION_MODEL_PATH = os.path.join(PREDICTION_MODELS_ROOT_PATH, MODEL_TYPE, PREDICTION_MODEL_FILENAME)

IMG_HEIGHT = 96
IMG_WIDTH = 96
COLOR_MODE = "grayscale"

DIM = 4 if COLOR_MODE == "rgba" else 1 if COLOR_MODE == "grayscale" else 3

# setting
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    num_faces=1, 
    base_options=BaseOptions(model_asset_path=DETECTION_MODEL_PATH), 
    running_mode=VisionRunningMode.IMAGE
)

#Create instance of FaceLandmarker  
landmarker = FaceLandmarker.create_from_options(options)

#Capture image from cam
cap = cv2.VideoCapture(CAMERA_DEVICE_NUMBER)

#YOLO model
model = YOLO(PREDICTION_MODEL_PATH)

total_time = 0
total_frames = 0

while cap.isOpened():
    flag = True

    time_start = time.time()
    success, frame = cap.read() # read frame

    if not success:
        print('Can not get Frame')
        break
    
    H, W, C = frame.shape

    rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_landmarker_result = landmarker.detect(rgb_image) # detect landmarks(need use mediapipe type image)
    
    face_landmarks_list = face_landmarker_result.face_landmarks

    if not face_landmarks_list:
        flag = False
    
    #---------------TEST GET EYE FEATURE--------------------------
    if flag:
        left_eye = []
        left_eye_img = []

        face_landmarks_list_len = len(face_landmarks_list)

        for idx in range(face_landmarks_list_len):
            face_landmarks = face_landmarks_list[idx]
            left_eye = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS]
            left_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in left_eye]
            
            #------------------Test fixed-size img---------------
            fixed_size_x = 32 #25
            fixed_size_y = 32 #10
            left_eye_mid = face_landmarks[473]
            left_eye_mid_pt_x = int(left_eye_mid.x*W)
            left_eye_mid_pt_y = int(left_eye_mid.y*H)

            left_eye_img = frame[left_eye_mid_pt_y-fixed_size_y:left_eye_mid_pt_y+fixed_size_y+1, left_eye_mid_pt_x-fixed_size_x:left_eye_mid_pt_x+fixed_size_x+1]
                
            right_eye = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS]
            right_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in right_eye]
            right_x, right_y, right_w, right_h = cv2.boundingRect(np.array(right_eye_pts))

        data = [left_eye_img, left_eye_img]

        results = model.predict(data)
    
        for result in results:
            result.summary()

    time_end = time.time()
    time_delta = time_end - time_start

    total_time = total_time + time_delta
    total_frames = total_frames + 1

    fps = total_frames / total_time

    #Average process time on rtx 3050 laptop ~= 0.03 sec
    print(f"Process time of current frame: {time_delta} secs")
    print(f"Current FPS: {fps}")
    
    # get frame after do landmark
    cv2.imshow("Face landmarks", frame)
    if cv2.waitKey(1) == 27: # press ESC to leave
        break
  
cap.release()
cv2.destroyAllWindows()