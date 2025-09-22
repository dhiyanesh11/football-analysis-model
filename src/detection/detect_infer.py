import cv2
from ultralytics import YOLO

# Load a bigger model for better accuracy (GPU can handle it)
model = YOLO("yolov8m.pt")   # try yolov8m.pt or yolov8l.pt

cap = cv2.VideoCapture("match.mp4")  # your football video

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:   # skip every 2nd frame (optional for speed)
        continue

    # Run detection on GPU (device=0)
    results = model.predict(frame, imgsz=1920, conf=0.25, device=0, verbose=False)

    # Draw boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Football Detection (GPU)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
