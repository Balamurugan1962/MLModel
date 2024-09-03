import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ConvNeXtXLarge
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
# Ensure GPU is being used
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU")
else:
    print("GPU not available, using CPU")
IMG_DIM=224
# Define model
pretrained = ConvNeXtXLarge(input_shape=(IMG_DIM, IMG_DIM, 3), include_top=False)
pretrained.trainable = False

model = Sequential([
    pretrained,
    GlobalAveragePooling2D(),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load weights
MODEL_PATH = "C:/Users/Duraisamy/Downloads/convnext_xlarge.weights.h5"
try:
    model.load_weights(MODEL_PATH)
    print("Weights loaded successfully")
except Exception as e:
    print(f"Error loading weights: {e}")


class_labels = ['Female', 'Male']  # Adjust based on your model's labels

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change index if necessary (0 is usually the default webcam)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_DIM, IMG_DIM))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Display the prediction
    label = class_labels[predicted_class]
    text = f"{label}: {confidence:.2f}"

    # Draw the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Gender Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

