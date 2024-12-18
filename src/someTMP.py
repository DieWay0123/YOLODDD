'''
#----------------------------------test MSR(Retinex光照補強)-------------------------------------
scales = [3, 5, 9]
b_gray, g_gray, r_gray = cv2.split(frame)
b_gray = MSR(b_gray, scales)
g_gray = MSR(g_gray, scales)
r_gray = MSR(r_gray, scales)
result = cv2.merge([b_gray, g_gray, r_gray])
cv2.imwrite("MSR_image.jpg", result)
#-----------------------------------------------------------------------------------------------
'''

#EYES OTSU
'''
    #-------------------------cv2 image process----------------------------
    if len(left_eye_img)!=0: #判斷是否有偵測到眼睛
        #!FIXME
        left_eye_img_o = cv2.imread('left_eye.jpg')
        #left_eye_img = Image.fromarray(left_eye_img)
        left_eye_img_gray = cv2.cvtColor(left_eye_img_o, cv2.COLOR_BGR2GRAY)
        left_eye_img_gray_blur = cv2.GaussianBlur(left_eye_img_gray, (3,3), 0)
        threshold, left_eye_img_otsu  = cv2.threshold(
            left_eye_img_gray_blur,
            0,
            255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        
        #cv2.imwrite('left_eye_otsu.jpg', left_eye_img_otsu)
    #-----------------------------------------------------------------------
'''

#DRAW LANDMARKS(EYES)
'''
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # get landmark use 
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])  
    
    
        connections = frozenset().union(*[
        mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
        mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
        ])
        solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        connections=connections,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
'''