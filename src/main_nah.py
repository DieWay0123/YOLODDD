from models import DrowsyDetecter

from ultralytics import YOLO

import mediapipe as mp
import cv2

import os


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

dd = DrowsyDetecter(landmarker, model, model, cap)
dd.run()