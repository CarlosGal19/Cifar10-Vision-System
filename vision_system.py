import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r"C:\Users\Carlos Galindo\Desktop\first_partial_exam_AI\model.h5")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No able to access to camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: No able to capture the frame.")
        break

    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(color_frame, (32, 32))

    img_normalized = img_resized / 255.0

    img_expanded = np.expand_dims(img_normalized, axis=0)

    pred = model.predict(img_expanded)

    predicted_class = classes[np.argmax(pred)]

    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real Time Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
