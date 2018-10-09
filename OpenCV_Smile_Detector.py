# CV Smile Detector

# Importing needed libraries
import cv2
import numpy as np
import sys

# Loading the cascades
face_cascade = cv2.CascadeClassifier('C:/Users/Olar/Documents/Computer Vision/Computer_Vision_A_Z_Template_Folder/Module 1 - Face Recognition/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('C:/Users/Olar/Documents/Computer Vision/Computer_Vision_A_Z_Template_Folder/Module 1 - Face Recognition/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640)  
cap.set(4,480)

sF = 1.05

while True:
    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor = sF, minNeighbors = 8, minSize = (55,55), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor = 1.7, minNeighbors = 22, minSize = (25,25), flags = cv2.CASCADE_SCALE_IMAGE)
        
        # Set region of interest for smiles 
        for(x,y,w,h) in smile:
        
            cv2.rectangle(roi_color, (x,y), (x+w, y+h), (0, 255, 0), 2)
            
    cv2.imshow('Smile Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
cap.release()
cv2.destroyAllWindows()