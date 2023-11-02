import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

confidence = 0.8
save = True
blurThreshold = 39

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

            # ----------------- check score--------------------
            if score >= confidence:

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
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)



                # ---------------- Normalize Value -----------------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingNumber), round(yc / ih, floatingNumber)
                wn, hn = round(w / iw, floatingNumber), round(h / ih, floatingNumber)

                # ---------------- avoid above 1 -----------------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                # ---------------- draw -----------------
                cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                   scale=1, thickness=1)

        if save:
            if all(listBlur) and listBlur!=[]:
                pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)
