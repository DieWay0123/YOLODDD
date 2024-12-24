from typing import Callable
from ultralytics import YOLO

from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import cv2
import time

import mediapipe as mp


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class Color(Enum):
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)


class DetectionInfo(Enum):
    FACE_NOT_DETECTED = "Face not detected"
    LEFT_EYE_NOT_DETECTED = "Left eye not detected"
    RIGHT_EYE_NOT_DETECTED = "Right eye not detected"
    MOUTH_NOT_DETECTED = "Mouth not detected"


class FrameWriter(object):
    def __init__(self, initial_x: int, initial_y: int):
        self._initial_x = initial_x
        self._initial_y = initial_y
        self._x = initial_x
        self._y = initial_y

    @property
    def x(self) -> int:
        return self._x
    
    @property
    def y(self) -> int:
        return self._y
    
    def reset_cursor(self) -> None:
        self._x = self._initial_x
        self._y = self._initial_y

    def move_cursor(self, direction: Direction, value: int) -> None:
        if direction == Direction.UP:
            self._y -= value
        elif direction == Direction.DOWN:
            self._y += value
        elif direction == Direction.LEFT:
            self._x -= value
        elif direction == Direction.RIGHT:
            self._x += value

    def write_text(self, frame: cv2.typing.MatLike, text: str, font: int, font_scale: float, color: Color, thickness: float, line_type: int):
        cv2.putText(frame, text, (self._x, self._y), font, font_scale, color.value, thickness, line_type)


class DrowsyDetector(object):
    EYES_CLOSED_FRAME_THRES = 30
    YAWN_FRAME_THRES = 60
    YAWN_COUNT_THRES = 10

    def __init__(
        self, 
        camera: cv2.VideoCapture, 
        face_detection_model, 
        left_eye_landmarks: list[int], 
        left_eye_kp_l: int, 
        left_eye_kp_r: int, 
        left_eye_kp_mid: int,
        right_eye_landmarks: list[int], 
        right_eye_kp_l: int, 
        right_eye_kp_r: int, 
        right_eye_kp_mid: int,
        eyes_offset: float, 
        mouth_landmarks: list[int], 
        mouth_kp_l: int, 
        mouth_kp_r: int, 
        mouth_offset: float, 
        eyes_classification_model: YOLO, 
        eyes_closed_label: str, 
        eyes_open_label: str, 
        mouth_classification_model: YOLO, 
        mouth_closed_label: str, 
        mouth_open_label: str
    ):
        '''
        camera: 攝影機物件
        face_detection_model: Mediapipe的臉部偵測模型
        left_eye_landmarks: 左眼節點編號列表
        left_eye_kp_l: 左眼最左側的節點編號
        left_eye_kp_r: 左眼最右側的節點編號
        left_eye_kp_mid: 左眼中心的節點編號
        left_eye_kp_l: 右眼最左側的節點編號
        left_eye_kp_r: 右眼最右側的節點編號
        left_eye_kp_mid: 右眼中心的節點編號
        eyes_offset: 裁切眼睛圖片用的
        mouth_kp_l: 嘴巴最左側的節點編號
        mouth_kp_r: 嘴巴最右側的節點編號
        mouth_offset: 裁切嘴巴圖片用的
        eyes_classification_model: 辨識眼睛開閉的YOLO模型, 
        eyes_closed_label: 眼睛閉上的label, 
        eyes_open_label: 眼睛睜開的label, 
        mouth_classification_model: 辨識嘴巴開閉的YOLO模型, 
        mouth_closed_label: 嘴巴閉上的label, 
        mouth_open_label: 嘴巴張開的label
        '''

        self._camera = camera
        self._face_detection_model = face_detection_model

        self._left_eye_landmarks = left_eye_landmarks
        self._left_eye_kp_l = left_eye_kp_l
        self._left_eye_kp_r = left_eye_kp_r
        self._left_eye_kp_mid = left_eye_kp_mid

        self._right_eye_landmarks = right_eye_landmarks
        self._right_eye_kp_l = right_eye_kp_l
        self._right_eye_kp_r = right_eye_kp_r
        self._right_eye_kp_mid = right_eye_kp_mid

        self._eyes_offset = eyes_offset

        self._mouth_landmarks = mouth_landmarks
        self._mouth_kp_l = mouth_kp_l
        self._mouth_kp_r = mouth_kp_r
        self._mouth_offset = mouth_offset

        self._eyes_classification_model = eyes_classification_model
        self._eyes_closed_label = eyes_closed_label
        self._eyes_open_label = eyes_open_label

        self._mouth_classification_model = mouth_classification_model
        self._mouth_closed_label = mouth_closed_label
        self._mouth_open_label = mouth_open_label

        #偵測到的臉
        self._face = None

        #裁切後的眼睛和嘴巴圖片
        self._left_eye_image: cv2.typing.MatLike | None = None
        self._right_eye_image: cv2.typing.MatLike | None = None
        self._mouth_image: cv2.typing.MatLike | None = None

        #原始幀和輸出幀
        self._frame: cv2.typing.MatLike | None = None
        self._output_frame: cv2.typing.MatLike | None = None

    def _func_exec(self, func: Callable, *args, **kwargs):
        func(*args, **kwargs)

    @property
    def frame_captured(self) -> bool:
        return True if self._frame else False

    @property
    def face_detected(self) -> bool:
        return True if self._face else False
    
    @property
    def left_eye_detected(self) -> bool:
        return True if self._left_eye_image is not None and self._left_eye_image.size > 0 else False
    
    @property
    def right_eye_detected(self) -> bool:
        return True if self._right_eye_image is not None and self._right_eye_image.size > 0 else False
    
    @property
    def mouth_detected(self) -> bool:
        return True if self._mouth_image is not None and self._mouth_image.size > 0 else False
    
    def _update_frame(self, frame: cv2.typing.MatLike):
        self._frame = frame
        self._output_frame = frame.copy()

    def _reset_frames(self) -> None:
        self._frame = None
        self._output_frame = None
    
    def _update_detected_face(self) -> None:
        rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._frame)
        face_landmarker_result = self._face_detection_model.detect(rgb_image)
        face_landmarks_list = face_landmarker_result.face_landmarks
        if face_landmarks_list:
            self._face = face_landmarks_list[0]
        else:
            self._face = None

    def _update_left_eye_image(self) -> None:
        H, W, _ = self._frame.shape

        left_eye = [self._face[i] for i in self._left_eye_landmarks]

        xl_idx = self._left_eye_landmarks.index(self._left_eye_kp_l)
        xr_idx = self._left_eye_landmarks.index(self._left_eye_kp_r)

        xl = int(left_eye[xl_idx].x * W)
        xr = int(left_eye[xr_idx].x * W)

        x = y = int(abs(xl - xr) / self._eyes_offset)
        mid = self._face[self._left_eye_kp_mid]
        mid_pt_x = int(mid.x * W)
        mid_pt_y = int(mid.y * H)

        self._left_eye_image = self._frame[mid_pt_y-y:mid_pt_y+x+1, mid_pt_x-x:mid_pt_x+x+1]

        cv2.circle(self._output_frame, (mid_pt_x, mid_pt_y), 5, Color.CYAN.value, 2)
        cv2.rectangle(self._output_frame, (mid_pt_x-x, mid_pt_y-y), (mid_pt_x+x+1, mid_pt_y+y+1), Color.GREEN.value, 2)
    
    def _update_right_eye_image(self) -> None:
        H, W, _ = self._frame.shape

        right_eye = [self._face[i] for i in self._right_eye_landmarks]

        xl_idx = self._right_eye_landmarks.index(self._right_eye_kp_l)
        xr_idx = self._right_eye_landmarks.index(self._right_eye_kp_r)

        xl = int(right_eye[xl_idx].x * W)
        xr = int(right_eye[xr_idx].x * W)

        x = y = int(abs(xl - xr) / self._eyes_offset)
        mid = self._face[self._right_eye_kp_mid]
        mid_pt_x = int(mid.x * W)
        mid_pt_y = int(mid.y * H)

        self._right_eye_image = self._frame[mid_pt_y-y:mid_pt_y+x+1, mid_pt_x-x:mid_pt_x+x+1]

        cv2.circle(self._output_frame, (mid_pt_x, mid_pt_y), 5, Color.CYAN.value, 2)
        cv2.rectangle(self._output_frame, (mid_pt_x-x, mid_pt_y-y), (mid_pt_x+x+1, mid_pt_y+y+1), Color.GREEN.value, 2)
    
    def _update_mouth_image(self) -> None:
        H, W, _ = self._frame.shape

        mouth = [self._face[i] for i in self._mouth_landmarks]

        xl_idx = self._mouth_landmarks.index(self._mouth_kp_l)
        xr_idx = self._mouth_landmarks.index(self._mouth_kp_r)

        xl = int(mouth[xl_idx].x * W)
        xr = int(mouth[xr_idx].x * W)
        yl = int(mouth[xl_idx].y * H)
        yr = int(mouth[xr_idx].y * H)

        x = y = int(abs(xl - xr) / self._mouth_offset)
        mid_pt_x = int((xl + xr) / 2)
        mid_pt_y = int((yl + yr) / 2)

        self._mouth_image = self._frame[mid_pt_y-y:mid_pt_y+x+1, mid_pt_x-x:mid_pt_x+x+1]

        cv2.circle(self._output_frame, (mid_pt_x, mid_pt_y), 5, Color.CYAN.value, 2)
        cv2.rectangle(self._output_frame, (mid_pt_x-x, mid_pt_y-y), (mid_pt_x+x+1, mid_pt_y+y+1), Color.GREEN.value, 2)

    def run(self) -> None:
        frame_writer = FrameWriter(10, 10)

        eyes_closed_frames = 0
        yawn_frames = 0

        yawn_count = 0

        eyes_closed_long = False
        yawn_long = False

        #計算花了幾秒的時間和處理了幾幀
        total_secs = 0
        total_frames = 0

        #相機是否開啟
        while self._camera.isOpened():
            #重設幀繪製器的座標
            frame_writer.reset_cursor()

            #開始計時
            time_start = time.time()

            #檢測是否有捕捉到幀
            success, frame = self._camera.read()
            if not success:
                break

            #更新幀
            self._update_frame(frame)

            #取得臉部的位置並更新
            self._update_detected_face()

            #如果有偵測到臉
            if self.face_detected:

                #打包更新五官圖片的函式然後用多線程加速
                funcs = [self._update_left_eye_image, self._update_right_eye_image, self._update_mouth_image]
                with ThreadPoolExecutor(max_workers=3) as pool:
                    pool.map(self._func_exec, funcs)

                #如果左右眼和嘴巴都有被偵測到
                if self.left_eye_detected and self.right_eye_detected and self.mouth_detected:

                    eyes_data = [self._left_eye_image, self._right_eye_image]
                    mouth_data = [self._mouth_image]

                    eyes_results = self._eyes_classification_model.predict(eyes_data)
                    mouth_results = self._mouth_classification_model.predict(mouth_data)

                    left_eye_summary = eyes_results[0].summary()
                    right_eye_summary = eyes_results[1].summary()
                    mouth_summary = mouth_results[0].summary()

                    summaries = [left_eye_summary, right_eye_summary, mouth_summary]
                    labels = ["Left Eye", "Right Eye", "Mouth"]

                    left_eye_closed = False
                    right_eye_closed = False
                    yawn = False

                    for idx, zipped in enumerate(zip(summaries, labels)):
                        summary, label = zipped

                        name = summary[0]['name']
                        confidence = summary[0]['confidence']

                        text = f'{label}: {name} ({confidence})'
                        frame_writer.move_cursor(Direction.DOWN, 40)
                        frame_writer.write_text(self._output_frame, text, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)
                        
                        #左眼
                        if idx == 0:
                            if name == self._eyes_closed_label:
                                left_eye_closed = True
                        #右眼
                        elif idx == 1:
                            if name == self._eyes_closed_label:
                                right_eye_closed = True
                        #嘴巴
                        elif idx == 2:
                            if name == self._mouth_open_label:
                                yawn = True

                    #兩眼都閉
                    if left_eye_closed and right_eye_closed:
                        eyes_closed_frames += 1
                        if eyes_closed_frames >= self.EYES_CLOSED_FRAME_THRES and not eyes_closed_long:
                            eyes_closed_long = True
                    else:
                        eyes_closed_frames = 0
                        if eyes_closed_frames <= 0:
                            eyes_closed_long = False
                            
                    if eyes_closed_long:
                        text = "Beep! Beep! Beep!"
                        frame_writer.move_cursor(Direction.DOWN, 40)
                        frame_writer.write_text(self._output_frame, text, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)
                    if yawn:
                        yawn_frames += 1
                        if yawn_frames >= self.YAWN_FRAME_THRES and not yawn_long:
                            yawn_count += 1
                            yawn_long = True
                    else:
                        yawn_frames = 0
                        yawn_long = False

                    #打哈欠的次數太多
                    if yawn_count >= self.YAWN_COUNT_THRES:
                        text = "You are drowsy"
                        frame_writer.move_cursor(Direction.DOWN, 40)
                        frame_writer.write_text(self._output_frame, text, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)
                else:
                    #左眼未被偵測到
                    if not self.left_eye_detected:
                        frame_writer.move_cursor(Direction.DOWN, 40)
                        frame_writer.write_text(self._output_frame, DetectionInfo.LEFT_EYE_NOT_DETECTED.value, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)

                    #右眼未被偵測到
                    if not self.right_eye_detected:
                        frame_writer.move_cursor(Direction.DOWN, 40)
                        frame_writer.write_text(self._output_frame, DetectionInfo.RIGHT_EYE_NOT_DETECTED.value, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)

                    #嘴巴未被偵測到
                    if not self.mouth_detected:
                        frame_writer.move_cursor(Direction.DOWN, 40)
                        frame_writer.write_text(self._output_frame, DetectionInfo.MOUTH_NOT_DETECTED.value, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)
            else:
                frame_writer.move_cursor(Direction.DOWN, 40)
                frame_writer.write_text(self._output_frame, DetectionInfo.FACE_NOT_DETECTED.value, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)

            time_end = time.time()
            time_delta = time_end - time_start

            total_secs += time_delta
            total_frames += 1
            fps = int(total_frames / total_secs)

            text = f"Yawn count: {yawn_count}"
            frame_writer.move_cursor(Direction.DOWN, 40)
            frame_writer.write_text(self._output_frame, text, cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2, 1)

            text = f"FPS: {fps}"
            frame_writer.move_cursor(Direction.DOWN, 40)
            frame_writer.write_text(self._output_frame, text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Color.BLUE, 1, 2)

            cv2.imshow("DROWSY DETECTION", self._output_frame)
            if cv2.waitKey(1) == 27: # press ESC to leave
                break

        print(f"End after {total_secs} secs")
        print(f"Captured frames: {total_frames}")
