import pickle
import cv2
import pyttsx3
import threading
from gtts import gTTS
from playsound import playsound
import os
import uuid
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#-------------------speak------------

def speak(text):
    def _speak():
        try:
            tts = gTTS(text=text, lang='ta')  # 'ta' = Tamil
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Speech error:", e)
    
    threading.Thread(target=_speak, daemon=True).start()
#----------------------------------------------------------

# Tamil Labels Dictionary
labels_dict = {
     26: 'வணக்கம்', 27: 'முடிந்தது', 
    28: 'நன்றி', 29: 'விருப்பமானவை', 30: 'மன்னிக்கவும்', 31: 'தயவுசெய்து', 32: 'நீங்கள் வரவேற்கப்படுகிறீர்கள்'
}

# Load Tamil font (ensure Latha.ttf is in the same directory or provide a correct path)
font_path = "Latha.ttf"  # Use a Tamil-supported font
font = ImageFont.truetype(font_path, 40)

last_prediction = ""
last_spoken_time = time.time()
cooldown = 1

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

    try:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        if predicted_character != last_prediction and time.time() - last_spoken_time > cooldown:
            print("Predicted character:", predicted_character)
            speak(predicted_character)
            last_prediction = predicted_character
            last_spoken_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Convert frame to PIL Image to support Tamil font rendering
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((x1, y1 - 50), predicted_character, font=font, fill=(0, 0, 0))

        # Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        pass

    cv2.imshow('frame', frame)

    # Exit when 'q' key is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
