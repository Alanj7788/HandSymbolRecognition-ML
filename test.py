import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller,Key
import math
import time
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui

# Initialize camera, detector, classifier, and other variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["I hope this letter finds you well", "Yours sincerely", "I look forward to hearing from you soon",
          "To whom it may concern"]

keyboard = Controller()
keys = [["Q", "W", "E", "R", "T", "Y", "U"],
        ["I", "O", "P", "A", "S", "D", "F"],
        ["G", "H", "J", "K", "L", "Z", "X"],
        ["C", "V", "B", "N", "M", ",", "<-"],
        ["/", "<-'", ".", " "]]

finalText = ""
last_capture_time = time.time()
last_prediction = None
gesture_prediction_active = False  # Initially set to False (off)
keyboard_detection_active = False  # Initially set to False (off)

# Tkinter setup
root = tk.Tk()
root.title("Hand Gesture and Virtual Keyboard Detection")

# Create frames for video and text
video_label = tk.Label(root)
video_label.pack()

word_label = tk.Label(root, text="", font=("Helvetica", 16))
word_label.pack()

# Button to toggle gesture prediction
toggle_gesture_button = tk.Button(root, text="Turn On Gesture Prediction", command=lambda: toggle_feature("gesture"))
toggle_gesture_button.pack()

# Button to toggle keyboard detection
toggle_keyboard_button = tk.Button(root, text="Turn On Keyboard Detection", command=lambda: toggle_feature("keyboard"))
toggle_keyboard_button.pack()

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 3) #letter size
    return img

class Button():
    def __init__(self, pos, text, size=[65, 65]): # 55 is square pink size
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([j * 73 + 120, 73 * i+40], key)) # +40 values are for margin from sides, *70 is for padding bw buttons

# Function to toggle features on/off
def toggle_feature(feature):
    global gesture_prediction_active, keyboard_detection_active
    if feature == "gesture":
        gesture_prediction_active = not gesture_prediction_active
        toggle_gesture_button.config(text="Turn Off Gesture Prediction" if gesture_prediction_active else "Turn On Gesture Prediction")
    elif feature == "keyboard":
        keyboard_detection_active = not keyboard_detection_active
        toggle_keyboard_button.config(text="Turn Off Keyboard Detection" if keyboard_detection_active else "Turn On Keyboard Detection")

# Function to handle hand gesture prediction
def perform_gesture_prediction(hands, img):
    global last_capture_time, last_prediction

    current_time = time.time()
    if current_time - last_capture_time >= 1.5:
        last_capture_time = current_time

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure the bounding box does not exceed image boundaries
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

            imgCrop = img[y1:y2, x1:x2]
            aspectRatio = h / w

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            last_prediction = labels[index]
            word_label.config(text=last_prediction)
        else:
            word_label.config(text="")
            if last_prediction:
                pyautogui.write(last_prediction, interval=0.1)
                last_prediction = None  # Reset after writing

# Function to handle keyboard detection
def perform_keyboard_detection(hands, img):
    global finalText

    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        bbox = hand1["bbox"]
        img = drawAll(img, buttonList)
        if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:  # lmList[8][x] x value and lmList[8][1] y value
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)  # highlight the key when index is on for pink color
                    cv2.putText(img, button.text, (x + 12, y + 50), cv2.FONT_HERSHEY_PLAIN, 2.15, (255, 255, 255), 4)  # highlight the key when index is on for letter white color

                    # Extract x, y values for landmarks 8 and 12
                    x1, y1 = lmList[5][0], lmList[5][1]
                    x2, y2 = lmList[4][0], lmList[4][1]

                    l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)  # get coordinates of landmarks 8 and 12
                    print(l)

                    # When Clicked
                    if l < 30:  # Adjust the threshold for click detection
                        if button.text == "<-'":
                            keyboard.press(Key.enter)
                            keyboard.release(Key.enter)
                        elif button.text == "<-":
                            keyboard.press(Key.backspace)
                            keyboard.release(Key.backspace)
                        else:
                            keyboard.press(button.text)
                            keyboard.release(button.text)

                            #cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            #cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            #finalText += button.text
                        sleep(0.7)

       # cv2.rectangle(img, (50, 350), (1032, 450), (175, 0, 175), cv2.FILLED)  # placeholder for displaying text
       # cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)  # placeholder for displaying text

# Function to update the video frame and perform gesture recognition or keyboard detection
def update_frame():
    global gesture_prediction_active, keyboard_detection_active

    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if gesture_prediction_active:
        perform_gesture_prediction(hands, img)
    if keyboard_detection_active:
        perform_keyboard_detection(hands, img)

    img = cv2.resize(img, (640, 480))  # Resize the video to 640x480
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
