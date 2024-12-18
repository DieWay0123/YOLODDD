from concurrent.futures import thread
from tkinter import Image
import PIL
import cv2
import numpy as np
import mediapipe as mp
import os
import math
from matplotlib import pyplot as plt

from ultralytics import YOLO
from threading import Thread


LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LANDMARKS = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
INNER_LIP_LANDMARKS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

CAMERA_DEVICE_NUMBER = 0

DETECTION_MODELS_ROOT_PATH = "d_models"
PREDICTION_MODELS_ROOT_PATH = "models"

MODEL_TYPE = "eyes"

DETECTION_MODEL_FILENAME = "face_landmarker.task"
PREDICTION_MODEL_FILENAME = "model-20241213-225015.pt"

PICTURE_ROOT_PATH = "pictures"
DETECTION_LEFT_EYE_PICTURE_FILENAME = "left_eye.png"
DETECTION_RIGHT_EYE_PICTURE_FILENAME = "right_eye.png"
DETECTION_MOUTH_PICTURE_FILENAME = "mouth.png"


#!FIX: Detection model都用同一個 沒有分部位
DETECTION_MODEL_PATH = os.path.join(DETECTION_MODELS_ROOT_PATH, DETECTION_MODEL_FILENAME)
PREDICTION_MODEL_PATH = os.path.join(PREDICTION_MODELS_ROOT_PATH, MODEL_TYPE, PREDICTION_MODEL_FILENAME)

#!ADD
DETECTION_LEFT_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_LEFT_EYE_PICTURE_FILENAME)
DETECTION_RIGHT_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_RIGHT_EYE_PICTURE_FILENAME)
DETECTION_MOUTH_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_MOUTH_PICTURE_FILENAME)

#!ADD
PREDICTION_EYE_PICTURE_FILENAME = "YOLO_eye_pred_"
PREDICTION_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, PREDICTION_EYE_PICTURE_FILENAME)

IMG_HEIGHT = 96
IMG_WIDTH = 96
COLOR_MODE = "grayscale"

DIM = 4 if COLOR_MODE == "rgba" else 1 if COLOR_MODE == "grayscale" else 3

def thread_safe_predict(model, data, save_path):
  results = model.predict(data)
  try:
    for i, result in enumerate(results):
      #0: left eye, 1: right eye
      r = result.plot()
      save_path = f"{save_path}_{i}.png"
      cv2.imwrite(save_path, r)
  except:
    pass

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
model = YOLO(PREDICTION_MODEL_PATH)

while cap.isOpened():
    success, frame = cap.read() # read frame
    frame_draw = frame.copy()
    if not success:
        print('Can not get Frame')
        break
    
    H, W, C = frame.shape
    rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_landmarker_result = landmarker.detect(rgb_image) # detect landmarks(need use mediapipe type image)
    
    face_landmarks_list = face_landmarker_result.face_landmarks
    
    #---------------TEST GET EYE FEATURE--------------------------
    left_eye = []
    left_eye_img = []
    flag = True

    face_landmarks_list_len = len(face_landmarks_list)

    for idx in range(face_landmarks_list_len):
        face_landmarks = face_landmarks_list[idx]
        left_eye = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS]
        left_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in left_eye]
        
        #!------------------------------------------------------Test fixed-size img---------------------------------------------------
        fixed_size_x = 39 #25
        fixed_size_y = 39 #10
        left_eye_mid = face_landmarks[473] #get the middle landmark of left eye
        left_eye_mid_pt_x = int(left_eye_mid.x*W)
        left_eye_mid_pt_y = int(left_eye_mid.y*H)
        left_eye_img = frame[left_eye_mid_pt_y-fixed_size_y:left_eye_mid_pt_y+fixed_size_y+1, left_eye_mid_pt_x-fixed_size_x:left_eye_mid_pt_x+fixed_size_x+1]
        
        right_eye_mid = face_landmarks[468]
        right_eye_mid_pt_x = int(right_eye_mid.x*W)
        right_eye_mid_pt_y = int(right_eye_mid.y*H)
        right_eye_img = frame[right_eye_mid_pt_y-fixed_size_y:right_eye_mid_pt_y+fixed_size_y+1, right_eye_mid_pt_x-fixed_size_x:right_eye_mid_pt_x+fixed_size_x+1]
        
        #right_eye = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS]
        #right_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in right_eye]
        #right_x, right_y, right_w, right_h = cv2.boundingRect(np.array(right_eye_pts))
        #right_eye_img = frame[right_y:right_y+right_h, right_x:right_x+right_w]
        
        #畫圖
        cv2.circle(frame_draw, (left_eye_mid_pt_x, left_eye_mid_pt_y), 5, (255, 255, 0), 2)
        cv2.circle(frame_draw, (right_eye_mid_pt_x, right_eye_mid_pt_y), 5, (255, 255, 0), 2)
        cv2.rectangle(frame_draw, (left_eye_mid_pt_x-fixed_size_x, left_eye_mid_pt_y-fixed_size_y), (left_eye_mid_pt_x+fixed_size_x+1, left_eye_mid_pt_y+fixed_size_y+1), (0, 255, 0), 2)
        cv2.rectangle(frame_draw, (right_eye_mid_pt_x-fixed_size_x, right_eye_mid_pt_y-fixed_size_y), (right_eye_mid_pt_x+fixed_size_x+1, right_eye_mid_pt_y+fixed_size_y+1), (0, 255, 0), 2)
        #!------------------------------------------------------Test fixed-size img---------------------------------------------------

        #---------------------------------------------------------mouth img---------------------------------------------------------
        MOUTH_POINT = [13, 308, 14, 78]#上右下左順序
        mouth_pts = [face_landmarks[i] for i in MOUTH_POINT]
        for point in mouth_pts:
          point.x *= W
          point.y *= H
        
        the_top_mouth_pt = mouth_pts[0]
        the_right_mouth_pt = mouth_pts[1]
        the_bottom_mouth_pt = mouth_pts[2]
        the_left_mouth_pt = mouth_pts[3]
        
        mouth_height = math.sqrt((the_top_mouth_pt.x-the_bottom_mouth_pt.x)**2+(the_top_mouth_pt.y-the_bottom_mouth_pt.y)**2)
        mouth_width = math.sqrt((the_left_mouth_pt.x-the_right_mouth_pt.x)**2+(the_left_mouth_pt.y-the_right_mouth_pt.y)**2)
        
        mouth_middle_pt = (the_left_mouth_pt.x+mouth_width/2, the_top_mouth_pt.y+mouth_height/2)

        #畫圖        
        cv2.circle(frame_draw, (int(mouth_middle_pt[0]), int(mouth_middle_pt[1])), 5, (255, 255, 0), 2)
        cv2.rectangle(frame_draw, (int(mouth_middle_pt[0]-mouth_width), int(mouth_middle_pt[1]+mouth_height)), (int(mouth_middle_pt[0]+mouth_width), int(mouth_middle_pt[1]-mouth_height)), (0, 255, 0), 2)
        
        MOUTH_OPEN_THRESH = 20
        if mouth_height > MOUTH_OPEN_THRESH:
          mouth_img = frame[int(mouth_middle_pt[1]-mouth_height):int(mouth_middle_pt[1]+mouth_height), int(mouth_middle_pt[0]-mouth_width):int(mouth_middle_pt[0]+mouth_width)]
        else:
          mouth_img = frame[int(mouth_middle_pt[1]-mouth_width):int(mouth_middle_pt[1]+mouth_width*1.5), int(mouth_middle_pt[0]-mouth_width):int(mouth_middle_pt[0]+mouth_width)]
        #---------------------------------------------------------mouth img---------------------------------------------------------
    
    #儲存眼部偵測圖片(test)
    try:
      cv2.imwrite(DETECTION_LEFT_EYE_PICTURE_PATH, left_eye_img)
      cv2.imwrite(DETECTION_RIGHT_EYE_PICTURE_PATH, right_eye_img)
    except:
      print("ERROR SAVE IMG")
      pass
    
    #儲存嘴巴偵測圖片(test)
    try:
      cv2.imwrite(DETECTION_MOUTH_PICTURE_PATH, mouth_img)
    except:
      pass
    
    try:
      #left_eye_img_gs = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
      #!FIX Predict改使用thread
      eyes_data = [left_eye_img, right_eye_img]
      Thread(target=thread_safe_predict, args=(model, eyes_data, PREDICTION_EYE_PICTURE_PATH)).start()
    except:
        pass

    # get frame after do landmark
    cv2.imshow("Face landmarks", frame_draw)
    if cv2.waitKey(1) == 27: # press ESC to leave
        break
cap.release()
cv2.destroyAllWindows()