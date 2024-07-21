# Virtual Touch

This project demonstrates a system that uses hand gestures for various tasks such as virtual keyboard input, predefined words insertion, and virtual mouse control. The project is implemented using Python with OpenCV, cvzone, and other supporting libraries.

## Features

- **Hand Gesture Recognition:** Recognizes specific hand gestures to insert predefined words.
- **Virtual Keyboard:** Allows typing on a virtual keyboard using hand movements.
- **Virtual Mouse Control:** Enables controlling the mouse cursor and performing actions like clicks and scrolls through hand gestures.

## Installation

1. Clone the repository and switch to the specific branch:
    ```bash
    git clone https://github.com/Alanj7788/HandSymbolRecognition-ML.git
    cd HandSymbolRecognition-ML
    git checkout ICT_final-submission
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python
    pip install cvzone
    pip install pynput
    pip install pillow
    pip install pyautogui
    ```

3. Download the pre-trained model and labels file, and place them in the `Model` directory:
    - [keras_model.h5](path-to-model)
    - [labels.txt](path-to-labels)

## Usage

1. Run the script:
    ```bash
    python test.py
    ```

2. The application window will open with three buttons to toggle different features:
    - **Gesture Prediction:** Recognizes predefined gestures and types corresponding text.
    - **Keyboard Detection:** Enables virtual keyboard for typing.
    - **Virtual Mouse:** Controls the mouse pointer and performs clicks and scrolls.

## Team Members

- **Alan Jose**
- **Akhil Jose**
- **Fathimathul Thabshira P J**
- **Josiah Benny**

## Project Structure

- **test.py:** Main script that runs the application.
- **Model/keras_model.h5:** Pre-trained model for hand gesture classification.
- **Model/labels.txt:** Labels for the hand gesture model.

## Model Training

The model was trained by uploading images to [Teachable Machine](https://teachablemachine.withgoogle.com/) and obtaining a Keras model from there.

## Project Website

For more details about the project, visit our [project website](https://alanj7788.wixsite.com/virtualtouch).

