import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
model=load_model("emotion_model.keras")
emotion_labels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

cap=cv.VideoCapture(0)

while True:
    ret, frame=cap.read()
    if not ret:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h, x:x+w]
        roi_gray=cv.resize(roi_gray,(48,48))
        roi=roi_gray.astype("float32")/255.0
        roi=roi.reshape(1,48,48,1)

        prediction=model.predict(roi)
        label=emotion_labels[np.argmax(prediction)]

        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(frame,label,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
    
    cv.imshow("Emotion Detection",frame)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()

