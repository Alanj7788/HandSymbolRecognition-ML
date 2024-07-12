import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import cvzone
from pynput.keyboard import Controller
from PIL import Image

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", "'"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""

keyboard = Controller()


class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([j * 100 + 50, 100 * i + 50], key))


def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


def start_tracking():
    global finalText
    finalText = ""
    frame_placeholder = st.empty()
    text_placeholder = st.empty()
    stop_signal = st.session_state['stop_signal']

    while cap.isOpened():
        if st.session_state['stop_signal']:
            break

        success, img = cap.read()
        if not success:
            break
        hands, img = detector.findHands(img)

        if hands:
            hand1 = hands[0]
            lmList = hand1["lmList"]
            bbox = hand1["bbox"]
            img = drawAll(img, buttonList)
            if lmList:
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                        cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                        x1, y1 = lmList[8][0], lmList[8][1]
                        x2, y2 = lmList[12][0], lmList[12][1]

                        l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)
                        print(l)

                        if l < 40:
                            keyboard.press(button.text)
                            cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255),
                                        4)
                            finalText += button.text
                            sleep(0.45)

            cv2.rectangle(img, (50, 350), (1032, 450), (175, 0, 175), cv2.FILLED)
            cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Display frame
        frame_placeholder.image(img_pil)

        # Display text
        text_placeholder.text(f"Detected Text: {finalText}")


def main():
    st.title("Hand Tracking Virtual Keyboard")

    if 'stop_signal' not in st.session_state:
        st.session_state['stop_signal'] = False

    start_button = st.button("Start Hand Tracking")
    stop_button = st.button("Stop Hand Tracking")

    if start_button:
        st.session_state['stop_signal'] = False
        start_tracking()

    if stop_button:
        st.session_state['stop_signal'] = True


if __name__ == "__main__":
    main()