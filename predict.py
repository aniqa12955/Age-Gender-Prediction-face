import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("age_gender_model.h5")

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

age_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "50+"]

img = cv2.imread("test.jpg")

if img is None:
    print("❌ Image not found!")
    exit()

(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

faces_found = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        faces_found += 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # ✅ FIX 1: BGR → RGB (model RGB pe train hua tha)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        age_pred, gender_pred = model.predict(face, verbose=0)
        age = age_labels[np.argmax(age_pred)]

        # ✅ FIX 2: Gender logic sahi kiya
        # UTKFace: 0=Male, 1=Female → sigmoid > 0.5 means Female
        gender_prob = float(gender_pred[0][0])
        gender = "Female" if gender_prob > 0.5 else "Male"

        label = f"{gender}, {age}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        print(f"Face {faces_found}: {label} (confidence: {gender_prob:.2f})")

print("Total faces detected:", faces_found)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()