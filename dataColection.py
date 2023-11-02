import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2


offsetPercentageW = 10
offsetPercentageH = 20
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


                # ----------------- add offset --------------------
                offsetW = offsetPercentageW / 100 * w
                x = x - int(offsetW)
                w = w + int(offsetW * 2)

                offsetH = offsetPercentageH / 100 * h
                y = y - int(offsetH * 3)
                h = h + int(offsetH * 3.5)

                # ---------------- avoid below 0 -----------------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # --------- extract blur  noice cancel------------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("face", imgFace)
                blurValue = cv2.Laplacian(imgFace, cv2.CV_64F).var()






                # ---------------- draw -----------------
                cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)


        # if save:
        #     if all(listBlur) and listBlur!=[]:
        #

    cv2.imshow("Image", img)
    cv2.waitKey(1)
