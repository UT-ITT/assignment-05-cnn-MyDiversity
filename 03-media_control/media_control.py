import cv2
import time
import numpy as np
import tensorflow as tf
from pynput.keyboard import Key, Controller

# model constants 

MODEL_PATH = "03-media_control\gesture_recognition.keras"

CONDITIONS = ['stop', 'no_gesture','like', 'dislike', 'peace']

IMG_SIZE = 64

# variable for gesture display text
current_display_text = 'No gesture'

# prediction variable
predicted_label = 'no_gesture'

# variables for prediction cooldown (to prevent spam)
last_action_time = 0
cooldown_duration = 0

# gesture mapping
GESTURES = {
    "stop": {"name": "START/STOP", "key": Key.media_play_pause, "cooldown": 2.0},
    "like": {"name": "VOLUME UP", "key": Key.media_volume_up, "cooldown": 0.5},
    "dislike": {"name": "VOLUME DOWN", "key": Key.media_volume_down, "cooldown": 0.5},
    "peace": {"name": "SKIP", "key": Key.media_next, "cooldown": 3.0}
}

# gesture region
x1,y1 = 100, 200
x2,y2 = 300, 450

# initialize Pynput controller
keyboard = Controller()

# load prediction model
model = tf.keras.models.load_model(MODEL_PATH)

# webcam Setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # mirror frame horizontally to display
    frame = cv2.flip(frame, 1)

    # show gesture region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 

    # get cropped frame for gesture recognition
    crop = frame[y1:y2, x1:x2]

    # open window displaying the cropped frame
    cv2.imshow('crop', crop)

    resized_frame = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
    normalized_frame = resized_frame.astype(np.float32) / 255
    input_data = np.expand_dims(normalized_frame, axis = 0)

    # get prediction
    prediction = model.predict(input_data, verbose = 0)

    # only accept gesture with high confidence
    if (np.max(prediction[0])) > 0.9:
        print(np.max(prediction[0]))
        predicted_label = CONDITIONS[np.argmax(prediction[0])] 
        if predicted_label in GESTURES:
            current_display_text = f"Gesture: {GESTURES[predicted_label]['name']}"
        else:
            predicted_label = 'no_gesture'

    # current time
    current_time = time.time()

    # remaining cooldown
    cooldown_remaining = max(0.0, cooldown_duration - (current_time - last_action_time))

    # gesture handling
    if predicted_label in GESTURES:
        gesture = GESTURES[predicted_label]

        # only execute if cooldown time is over
        if (current_time - last_action_time) > cooldown_duration:
            keyboard.press(gesture['key'])
            keyboard.release(gesture['key'])
            print(f"Control sent: {gesture['name']}")
            last_action_time = current_time
            cooldown_duration = gesture['cooldown']


    # draw current gesture state
    if predicted_label in GESTURES:
        gesture_text = f"Gesture: {GESTURES[predicted_label]['name']}"
    else:
        gesture_text = "Gesture: no gesture"

    cv2.putText(frame, gesture_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (180, 0, 50), 2, cv2.LINE_AA
        )

    # draw cooldown status
    if cooldown_remaining > 0.0:
        cooldown_text = f"Cooldown: {cooldown_remaining:.1f}s"
    else:
        cooldown_text = "Ready"
    
    text_size = cv2.getTextSize(cooldown_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(frame, cooldown_text, ((frame.shape[1]-text_size[0]-10),30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 50), 2, cv2.LINE_AA)

    # show window
    cv2.imshow('Media Controller', frame)

    # quit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program stopped")