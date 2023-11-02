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
    # cv2.bilateralFilter(img, 5, 5, 6)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 5, 15)
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        # bboxInfo = 'id', 'bbox', 'score', 'center'
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            score = bbox["score"][0]
            # print(x, y, w, h)

            # ---------------- draw -----------------
            cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)

        # if save:
        #     if all(listBlur) and listBlur!=[]:
        #

    cv2.imshow("Image", img)
    cv2.waitKey(1)
