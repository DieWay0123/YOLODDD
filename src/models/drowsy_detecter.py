from ultralytics import YOLO

import cv2
import time
import mediapipe as mp


class PredictResult(object):
    def __init__(self):
        self._le_open = False
        self._re_open = False
        self._m_open = False


class SysInfo(object):
    def __init__(self):
        self._start_time = time.time()
        self._end_time = time.time()
        self._frames = 0
        self._secs = 0

    @property
    def time_delta(self) -> float:
        return self._end_time - self._start_time
    
    def update_st(self) -> None:
        self._start_time = time.time()

    def update_et(self) -> None:
        self._end_time = time.time()

    def frames_inc(self) -> None:
        self._frames = self._frames + 1

    def secs_inc(self) -> None:
        self._secs = self._secs + self.time_delta

    @property
    def fps(self) -> int:
        return int(self._secs / self._frames)


class DrowsyDetecter(object):
    LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
    LEFT_EYE_LANDMARKS_LEN = len(LEFT_EYE_LANDMARKS)

    def __init__(self, detector, eyes_predict_model: YOLO, mouth_predict_model: YOLO, cap: cv2.VideoCapture):
        self._detector = detector
        self._eyes_predict_model = eyes_predict_model
        self._mouth_predict_model = mouth_predict_model
        self._cap = cap

        #Reusable objects
        self._predict_result = PredictResult()
        self._sys_info = SysInfo()

        #States
        self._processing = False

    def collect_face_landmarker_result(self, frame: cv2.typing.MatLike) -> list:
        rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = self._detector.detect(rgb_image)

        return face_landmarker_result

    def get_left_eye_image(self, face_landmarks, frame: cv2.typing.MatLike) -> None:
        if not self._processing:
            return None
        
        H, W, C = frame.shape

        fixed_size_x = 32 #25
        fixed_size_y = 32 #10
        left_eye_mid = face_landmarks[473]
        left_eye_mid_pt_x = int(left_eye_mid.x * W)
        left_eye_mid_pt_y = int(left_eye_mid.y * H)

        left_eye_img = frame[left_eye_mid_pt_y-fixed_size_y:left_eye_mid_pt_y+fixed_size_y+1, left_eye_mid_pt_x-fixed_size_x:left_eye_mid_pt_x+fixed_size_x+1]
        
        return left_eye_img
    
    def get_right_eye_image(self, face_landmarks, frame: cv2.typing.MatLike) -> None:
        pass

    def get_mouth_image(self, face_landmarks, frame: cv2.typing.MatLike) -> None:
        pass

    def predict_eyes(self, left_eye_img: cv2.typing.MatLike, right_eye_image: cv2.typing.MatLike) -> None:
        #tuple[Left eye, Right eye]
        left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
        data = [left_eye_img, left_eye_img]

        results = self._eyes_predict_model.predict(data)
        for result in results:
            result.show()

    def predict_mouth(self, mouth_image: cv2.typing.MatLike) -> None:
        pass


    def start_processing(self) -> None:
        self._sys_info.update_st()

        self._processing = True

    def end_processing(self) -> None:
        self._sys_info.update_et()
        self._sys_info.frames_inc()
        self._sys_info.secs_inc()

        self._processing = False

    def run(self) -> None:
        while self._cap.isOpened():
            self._sys_info.update_st()
            success, frame = self._cap.read() # read frame
    
            if not success:
                print('Can not get Frame')
                break

            self.start_processing()

            face_landmarker_result = self.collect_face_landmarker_result(frame)
            face_landmarks_list = face_landmarker_result.face_landmarks
            face_landmarker = face_landmarks_list[0]

            left_eye_image = self.get_left_eye_image(face_landmarker, frame)
            right_eye_image = self.get_left_eye_image(face_landmarker, frame)

            self.predict_eyes(left_eye_image, right_eye_image)

            self.end_processing()

            print(f"Process time of current frame: {self._sys_info.time_delta} secs")
            print(f"Current FPS: {self._sys_info.fps}")

            cv2.imshow("Face landmarks", frame)
            if cv2.waitKey(1) == 27: # press ESC to leave
                break
  
        self._cap.release()
        cv2.destroyAllWindows()

