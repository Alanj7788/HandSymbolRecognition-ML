import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyautogui

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

labels = ["I hope this letter finds you well", "Yours sincerely", "I look forward to hearing from you soon",
          "To whom it may concern"]

last_capture_time = time.time()
last_prediction = None
hand_present = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    current_time = time.time()
    if current_time - last_capture_time >= 1.5:
        last_capture_time = current_time

        if hands:
            hand_present = True
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure the bounding box does not exceed image boundaries
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y1:y2, x1:x2]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                last_prediction = labels[index]

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                last_prediction = labels[index]

            cv2.imshow('imgCrop', imgCrop)
            cv2.imshow('imgWhite', imgWhite)

        elif hand_present:
            hand_present = False
            if last_prediction:
                pyautogui.write(last_prediction, interval=0.1)
                print("Hand disappeared. Writing:", last_prediction)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
