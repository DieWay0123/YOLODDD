from models import DrowsyDetector

import cv2
import os
import mediapipe as mp

from ultralytics import YOLO


LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LANDMARKS = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
MOUTH_LANDMARKS = [61, 291]

INNER_LIP_LANDMARKS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

CAMERA_DEVICE_NUMBER = 1

DETECTION_MODELS_ROOT_PATH = "d_models"
PREDICTION_MODELS_ROOT_PATH = "models"

MODEL_TYPE_EYE = "eyes"
MODEL_TYPE_MOUTH = "mouth"

DETECTION_MODEL_FILENAME = "face_landmarker.task"
EYES_PREDICTION_MODEL_FILENAME = "model-20241213-225015.pt"
MOUTH_PREDICTION_MODEL_FILENAME = "model-20241224-142512.pt"

PICTURE_ROOT_PATH = "pictures"
DETECTION_LEFT_EYE_PICTURE_FILENAME = "left_eye.png"
DETECTION_RIGHT_EYE_PICTURE_FILENAME = "right_eye.png"
DETECTION_MOUTH_PICTURE_FILENAME = "mouth.png"

IMAGE_SAVE_PATH = "saved_images"


#!FIX: Detection model都用同一個 沒有分部位
DETECTION_MODEL_PATH = os.path.join(DETECTION_MODELS_ROOT_PATH, DETECTION_MODEL_FILENAME)
EYES_PREDICTION_MODEL_PATH = os.path.join(PREDICTION_MODELS_ROOT_PATH, MODEL_TYPE_EYE, EYES_PREDICTION_MODEL_FILENAME)
MOUTH_PREDICTION_MODEL_PATH = os.path.join(PREDICTION_MODELS_ROOT_PATH, MODEL_TYPE_MOUTH, MOUTH_PREDICTION_MODEL_FILENAME)

#!ADD
DETECTION_LEFT_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_LEFT_EYE_PICTURE_FILENAME)
DETECTION_RIGHT_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_RIGHT_EYE_PICTURE_FILENAME)
DETECTION_MOUTH_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_MOUTH_PICTURE_FILENAME)

#!ADD
PREDICTION_EYE_PICTURE_FILENAME = "YOLO_eye_pred_"
PREDICTION_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, PREDICTION_EYE_PICTURE_FILENAME)

#Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 2
LINE_TYPE = 2

#Thres
YAWN_FRAME_THRES = 60
EYES_CLOSED_FRAME_THRES = 30
YAWN_COUNT_THRES = 10

# mediapipe臉部landmmark偵測 setting
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    num_faces=1, 
    base_options=BaseOptions(model_asset_path=DETECTION_MODEL_PATH), 
    running_mode=VisionRunningMode.IMAGE
)
# create instance of FaceLandmarker  
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_DEVICE_NUMBER)

eyes_model = YOLO(EYES_PREDICTION_MODEL_PATH)
mouth_model = YOLO(MOUTH_PREDICTION_MODEL_PATH)

eyes_model = eyes_model.cuda()
mouth_model = mouth_model.cuda()

detector = DrowsyDetector(
    cap, 
    landmarker, 
    LEFT_EYE_LANDMARKS, 
    263, 
    362, 
    473, 
    RIGHT_EYE_LANDMARKS, 
    33, 
    133, 
    468, 
    1.2, 
    MOUTH_LANDMARKS, 
    61, 
    291, 
    1.4, 
    eyes_model, 
    "close eyes", 
    "open eyes", 
    mouth_model, 
    "no yawn", 
    "yawn"
)

detector.run()

cap.release()
cv2.destroyAllWindows()