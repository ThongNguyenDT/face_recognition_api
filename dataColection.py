import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

camW, camH = 640, 480
floatingNumber = 6

cap = cv2.VideoCapture(0)
cap.set(3, camW)
cap.set(4, camH)
detector = FaceDetector()
while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
