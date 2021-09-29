import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read() # ret:check xem ok chưa, frame:khung hình

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # cvt: convert to
    faces = face_cascade.detectMultiScale(gray)

    for(x, y, w, h) in faces:
        # frame:hình ảnh từ webcam,(x, y):điểm đầu tiên của webcam,x+...
        # tịnh tiến trong không gian: w:width, h:height
        # (0, 0, 225):màu đỏ, 2: độ dày
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 2)

    cv2.imshow('DETECTING FACE', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
