import cv2
import numpy as np
import tensorflow as tf


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def detect_and_predict_face(frame, face_detector, model, class_names, target_size=(300, 300)):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = preprocess_image(face, target_size)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        preds = model.predict(np.vstack(faces))

    return locs, preds


def run_face_recognition(model_path, class_names):
    model = load_model(model_path)
    face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        locs, preds = detect_and_predict_face(frame, face_detector, model, class_names)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            max_prob = np.max(pred)
            class_id = np.argmax(pred)

            if max_prob < 0.5:
                label = "Unknown"
            else:
                label = class_names[class_id]

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, f"{label}: {max_prob:.2f}", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "model_efficientnet_b3.keras"
    class_names = ["person1", "person2", "person3"]  # Add your actual class names here
    run_face_recognition(model_path, class_names)
