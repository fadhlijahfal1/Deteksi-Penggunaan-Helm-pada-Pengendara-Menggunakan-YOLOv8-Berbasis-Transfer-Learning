import cv2
import math
import cvzone
from ultralytics import YOLO

# Kamera laptop (built-in webcam)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load model YOLO
model = YOLO("Weights/best.pt")

classNames = ['With Helmet', 'Without Helmet']

while True:
    success, img = cap.read()
    if not success:
        print("Kamera laptop tidak terbaca")
        break

    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            cvzone.putTextRect(
                img,
                f'{classNames[cls]} {conf}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1
            )

    cv2.imshow("Helmet Detection - Laptop Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()