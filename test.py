import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui

# Initialize camera, detector, classifier, and other variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["I hope this letter finds you well", "Yours sincerely", "I look forward to hearing from you soon",
          "To whom it may concern"]

last_capture_time = time.time()
last_prediction = None
gesture_prediction_active = False  # Initially set to False (off)
keyboard_frame_visible = False  # Initially set to False (hidden)

# Tkinter setup
root = tk.Tk()
root.title("Hand Gesture Detection")

# Create frames for video and text
video_label = tk.Label(root)
video_label.pack()

word_label = tk.Label(root, text="", font=("Helvetica", 16))
word_label.pack()

# Create a frame for the on-screen keyboard
keyboard_frame = tk.Frame(root)

# Function to handle key presses on the keyboard
def on_key_press(letter):
    current_text = word_label.cget("text")
    word_label.config(text=current_text + letter)

# Create on-screen keyboard buttons
keyboard_letters = [
    'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
    'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
    'Z', 'X', 'C', 'V', 'B', 'N', 'M'
]

# Arrange buttons in a grid
for i, letter in enumerate(keyboard_letters):
    button = tk.Button(keyboard_frame, text=letter, command=lambda l=letter: on_key_press(l))
    button.grid(row=i // 10, column=i % 10, padx=7, pady=7, ipadx=7, ipady=7)

# Function to toggle gesture prediction on/off
def toggle_gesture_prediction():
    global gesture_prediction_active
    gesture_prediction_active = not gesture_prediction_active
    if gesture_prediction_active:
        toggle_button.config(text="Turn Off Gesture Prediction")
    else:
        toggle_button.config(text="Turn On Gesture Prediction")

# Function to toggle visibility of the on-screen keyboard
def toggle_keyboard():
    global keyboard_frame_visible
    keyboard_frame_visible = not keyboard_frame_visible
    if keyboard_frame_visible:
        keyboard_frame.pack()
        keyboard_toggle_button.config(text="Hide Keyboard")
    else:
        keyboard_frame.pack_forget()
        keyboard_toggle_button.config(text="Show Keyboard")

# Button to toggle gesture prediction
toggle_button = tk.Button(root, text="Turn On Gesture Prediction", command=toggle_gesture_prediction)
toggle_button.pack()

# Button to toggle visibility of the on-screen keyboard
keyboard_toggle_button = tk.Button(root, text="Show Keyboard", command=toggle_keyboard)
keyboard_toggle_button.pack()

# Function to update the video frame and perform gesture recognition
def update_frame():
    global last_capture_time, last_prediction, gesture_prediction_active

    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    current_time = time.time()
    if current_time - last_capture_time >= 1.5:
        last_capture_time = current_time

        if hands and gesture_prediction_active:
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
            if last_prediction and gesture_prediction_active:
                pyautogui.write(last_prediction, interval=0.1)
                last_prediction = None  # Reset after writing

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
