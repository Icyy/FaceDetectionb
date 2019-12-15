import cv2 as c
import numpy as np

cap = c.VideoCapture(0)

fc = c.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

while(True):
    ret, frame = cap.read() 


    gray = c.cvtColor(frame, c.COLOR_BGR2GRAY)

    faces = fc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        rgray = gray[y:y+h, x:x+w]
        rcolor = frame[y:y+h, x:x+w]
        color = (255, 0,0)
        stroke = 2
        width = x+w
        height = y+h
        c.rectangle(frame, (x,y), (width, height), color, stroke)




    c.imshow("video", frame)
    if c.waitKey(20) & 0xFF == ord('q'):
        break



cap.release()
c.destroyAllWindows()

