import cv2
import mediapipe as mp
import os
import math
import time
import threading

import simpleaudio

from ultralytics import YOLO


def write_text(frame, text, corner, font, font_scale, font_color, thickness, line_type):
    cv2.putText(
        frame,
        text, 
        corner, 
        font, 
        font_scale,
        font_color,
        thickness,
        line_type
    )

LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LANDMARKS = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
INNER_LIP_LANDMARKS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

CAMERA_DEVICE_NUMBER = 1

DETECTION_MODELS_ROOT_PATH = "d_models"
PREDICTION_MODELS_ROOT_PATH = "models"

MODEL_TYPE_EYE = "eyes"
MODEL_TYPE_MOUTH = "mouth"

DETECTION_MODEL_FILENAME = "face_landmarker.task"
EYES_PREDICTION_MODEL_FILENAME = "model-20241213-225015.pt"
MOUTH_PREDICTION_MODEL_FILENAME = "model-20241218-132319.pt"

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

total_time = 0
total_frames = 0

eyes_closed_frames = 0
yawn_frames = 0

yawn_count = 0

eyes_closed_long = False
yawn_long = False

playing = False

wave_object = simpleaudio.WaveObject.from_wave_file('never_gonna_give_you_up.wav')

while cap.isOpened():
    time_start = time.time()

    face_detected = False

    left_eye_detected = False
    right_eye_detected = False
    mouth_detected = False

    corner = (10, 0)

    success, frame = cap.read() # read frame
    frame_draw = frame.copy()

    if not success:
        print('Can not get Frame')
        break
    
    H, W, C = frame.shape
    rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_landmarker_result = landmarker.detect(rgb_image) # detect landmarks(need use mediapipe type image)
    
    face_landmarks_list = face_landmarker_result.face_landmarks

    if face_landmarks_list:
       face_detected = True
    
    #---------------TEST GET EYE FEATURE--------------------------
    if face_detected:
        left_eye_img = []
        right_eye_img = []
        mouth_img = []

        for face_landmarks in face_landmarks_list:
            left_eye = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS]

            left_eye_xl = int(left_eye[LEFT_EYE_LANDMARKS.index(263)].x * W)
            left_eye_xr = int(left_eye[LEFT_EYE_LANDMARKS.index(362)].x * W)

            right_eye_xl = int(left_eye[RIGHT_EYE_LANDMARKS.index(33)].x * W)
            right_eye_xr = int(left_eye[RIGHT_EYE_LANDMARKS.index(133)].x * W)

            #left_eye_pts = [(int(pt.x * W), int(pt.y * H)) for pt in left_eye]
            
            #!------------------------------------------------------Test fixed-size img---------------------------------------------------
            fixed_size_x = fixed_size_y = int(abs(left_eye_xl - left_eye_xr) / 1.2)
            left_eye_mid = face_landmarks[473] #get the middle landmark of left eye
            left_eye_mid_pt_x = int(left_eye_mid.x*W)
            left_eye_mid_pt_y = int(left_eye_mid.y*H)
            left_eye_img = frame[left_eye_mid_pt_y-fixed_size_y:left_eye_mid_pt_y+fixed_size_y+1, left_eye_mid_pt_x-fixed_size_x:left_eye_mid_pt_x+fixed_size_x+1]

            fixed_size_x = fixed_size_y = int(abs(right_eye_xl - right_eye_xr) / 1.2)
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
            MOUTH_POINT = [13, 308, 14, 78, 61, 291]#上右下左順序
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

            mouth_xl = int(mouth_pts[MOUTH_POINT.index(61)].x)
            mouth_xr = int(mouth_pts[MOUTH_POINT.index(291)].x)
            fixed_size_x = fixed_size_y = int(abs(mouth_xl - mouth_xr) / 1.4)

            #畫圖        
            cv2.circle(frame_draw, (int(mouth_middle_pt[0]), int(mouth_middle_pt[1])), 5, (255, 255, 0), 2)
            cv2.circle(frame_draw, (int(mouth_pts[MOUTH_POINT.index(61)].x), int(mouth_pts[MOUTH_POINT.index(61)].y)), 5, (255, 255, 0), 2)
            cv2.circle(frame_draw, (int(mouth_pts[MOUTH_POINT.index(291)].x), int(mouth_pts[MOUTH_POINT.index(291)].y)), 5, (255, 255, 0), 2)
            #cv2.rectangle(frame_draw, (int(mouth_middle_pt[0]-mouth_width), int(mouth_middle_pt[1]+mouth_height)), (int(mouth_middle_pt[0]+mouth_width), int(mouth_middle_pt[1]-mouth_height)), (0, 255, 0), 2)
            cv2.rectangle(frame_draw, (int(mouth_middle_pt[0]-fixed_size_x), int(mouth_middle_pt[1]+fixed_size_y)), (int(mouth_middle_pt[0]+fixed_size_x), int(mouth_middle_pt[1]-fixed_size_y)), (0, 255, 0), 2)
            
            MOUTH_OPEN_THRESH = 20
            if mouth_height > MOUTH_OPEN_THRESH:
                mouth_img = frame[int(mouth_middle_pt[1]-fixed_size_y):int(mouth_middle_pt[1]+fixed_size_y), int(mouth_middle_pt[0]-fixed_size_x):int(mouth_middle_pt[0]+fixed_size_x)]
                #mouth_img = frame[int(mouth_middle_pt[1]-mouth_height):int(mouth_middle_pt[1]+mouth_height), int(mouth_middle_pt[0]-mouth_width):int(mouth_middle_pt[0]+mouth_width)]
            else:
                mouth_img = frame[int(mouth_middle_pt[1]-fixed_size_y):int(mouth_middle_pt[1]+fixed_size_y), int(mouth_middle_pt[0]-fixed_size_x):int(mouth_middle_pt[0]+fixed_size_x)]
                #mouth_img = frame[int(mouth_middle_pt[1]-mouth_width):int(mouth_middle_pt[1]+mouth_width*1.5), int(mouth_middle_pt[0]-mouth_width):int(mouth_middle_pt[0]+mouth_width)]
        #---------------------------------------------------------mouth img---------------------------------------------------------
        
        if left_eye_img.size > 0:
            left_eye_detected = True
        
        if right_eye_img.size > 0:
            right_eye_detected = True

        if mouth_img.size > 0:
            mouth_detected = True

        if left_eye_detected and right_eye_detected and mouth_detected:
            eyes_data = [left_eye_img, right_eye_img]  
            mouth_data = [mouth_img]

            font_color = (0, 255, 0)

            '''
            用ProcessPool平行處理
            但更慢了:(

            try:
                #left_eye_img_gs = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
                #!FIX Predict改使用thread

                models = [eyes_model, mouth_model]
                datas = [eyes_data, mouth_data]

                labels = ["eyes", "mouth"]

                save_paths = [IMAGE_SAVE_PATH, IMAGE_SAVE_PATH]

                with ProcessPoolExecutor(max_workers=2) as pool:
                    pool.map(predict, models, datas, labels, save_paths)
            except:
                pass
            '''

            eyes_results = eyes_model.predict(eyes_data)
            mouth_results = mouth_model.predict(mouth_data)

            left_eye_summary = eyes_results[0].summary()
            right_eye_summary = eyes_results[1].summary()
            mouth_summary = mouth_results[0].summary()

            summaries = [left_eye_summary, right_eye_summary, mouth_summary]
            labels = ["Left Eye", "Right Eye", "Mouth"]

            for summary, label in zip(summaries, labels):
                name = summary[0]['name']
                confidence = summary[0]['confidence']

                corner = (corner[0], corner[1] + 40)

                name = summary[0]['name']
                confidence = summary[0]['confidence']

                text = f'{label}: {name} ({confidence})'
                write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)

            left_eye_closed = False
            right_eye_closed = False
            yawn = False

            for idx, summary in enumerate(summaries):
                name = summary[0]['name']
                confidence = summary[0]['confidence']

                if idx == 0:
                    if name == "close eyes":
                        left_eye_closed = True
                elif idx == 1:
                    if name == "close eyes":
                        right_eye_closed = True
                elif idx == 2:
                    if name == "yawn":
                        yawn = True

            if left_eye_closed and right_eye_closed:
                eyes_closed_frames += 1
                if eyes_closed_frames >= EYES_CLOSED_FRAME_THRES and not eyes_closed_long:
                    eyes_closed_long = True
                    thread = threading.Thread(target=wave_object.play)
                    thread.start()
            else:
                eyes_closed_frames = 0
                if eyes_closed_frames <= 0:
                    eyes_closed_long = False
                    
            if eyes_closed_long:
                text = "Beep! Beep! Beep!"
                corner = (corner[0], corner[1] + 40)
                font_color = (0, 0, 255)
                write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)

            if yawn:
                yawn_frames += 1
                if yawn_frames >= YAWN_FRAME_THRES and not yawn_long:
                    yawn_count += 1
                    yawn_long = True
            else:
                yawn_frames = 0
                yawn_long = False

            if yawn_count >= YAWN_COUNT_THRES:
                text = "You are drowsy"
                corner = (corner[0], corner[1] + 40)
                font_color = (0, 0, 255)
                write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)
        else:
            font_color = (0, 0, 255)
            if not left_eye_detected:
                text = f"Left Eye not detected!"
                corner = (corner[0], corner[1] + 40)
                write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)
            if not right_eye_detected:
                text = f"Right Eye not detected!"
                corner = (corner[0], corner[1] + 40)
                write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)
            if not mouth_detected:
                text = f"Mouth not detected!"
                corner = (corner[0], corner[1] + 40)
                write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)
    else:
        font_color = (0, 0, 255)
        text = f"Face not detected!"
        corner = (corner[0], corner[1] + 40)
        write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)

    font_color = (255, 0, 0)
    text = f"Yawn count: {yawn_count}"
    corner = (corner[0], corner[1] + 40)
    write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)

    '''
    #儲存眼部偵測圖片(test)
    try:
        cv2.imwrite(DETECTION_LEFT_EYE_PICTURE_PATH, left_eye_img)
        cv2.imwrite(DETECTION_RIGHT_EYE_PICTURE_PATH, right_eye_img)
    except:
        print("ERROR SAVING EYES IMG")
    
    #儲存嘴巴偵測圖片(test)
    try:
        cv2.imwrite(DETECTION_MOUTH_PICTURE_PATH, mouth_img)
    except:
        print("ERROR SAVING MOUTH IMG")
        pass
    '''

    time_end = time.time()
    time_delta = time_end - time_start

    total_time = total_time + time_delta
    total_frames = total_frames + 1

    fps = int(total_frames / total_time)

    text = f"FPS: {fps}"
    corner = (corner[0], corner[1] + 40)
    font_color = (255, 0, 0)
    write_text(frame_draw, text, corner, FONT, FONT_SCALE, font_color, THICKNESS, LINE_TYPE)

    # get frame after do landmark
    cv2.imshow("Face landmarks", frame_draw)
    if cv2.waitKey(1) == 27: # press ESC to leave
        break

cap.release()
cv2.destroyAllWindows()