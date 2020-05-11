#MachNEase
#Author : Namitha Guruprasad
#LinkedIn : linkedin.com/in/namitha-guruprasad-216362155
#Import important libraries
import dlib
import cv2
import os

#Webcam object
video_capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector() #For detecting faces
landmark_path="shape_predictor_68_face_landmarks.dat" #Path of the file - if stored in the same directory. Else, give the relative path
predictor = dlib.shape_predictor(landmark_path) #For identifying landmarks

while True:
    _ , img_frame = video_capture.read()
    gray_img = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    #clahe - Contrast Limited Adaptive Histogram Equalization - to avoid over - brightness, contrast limitation and amplification of noise
    #image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV)
    #If histogram bin is above the specified contrast limit - pixels are clipped and distributed uniformly
    #Here bi-polar interpolation is implemented
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#Arguments are optional 

    clahe_image = clahe.apply(gray_img)
    #Detecting faces in image
    detect = detector(clahe_image, 1) 
    for a,b in enumerate(detect):
        #For every detected face
        shape = predictor(clahe_image, b)
        #Obtaining coordinates
        for i in range(1,68):
            #68 landmark points on each face
            cv2.circle(img_frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,255), thickness=3)
            #Drawing the landmarks with thickness 3 - yellow in color

    cv2.imshow("image", img_frame)
    #Display the image frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Keyboard interrupt - 'q'
        break

