import cv2
from matplotlib.pyplot import gray

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_detection = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    
    # Blur faces
    for (x, y, w, h) in face_detection:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y:y+h, x:x+w] = face_roi
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


#show webcam 
    cv2.imshow('Face Blur Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
