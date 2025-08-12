from ultralytics import YOLO
import cv2
import math

# YOLO model load
model = YOLO("yolov8n.pt")

# Lanes ka input
lane_count = int(input("Enter number of lanes: "))
avg_time_per_vehicle = 10  # seconds per vehicle per lane

# Video load
cap = cv2.VideoCapture("traffic_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Variables
counted_ids = set()
offset = 10  # tolerance for line crossing

cv2.namedWindow("Traffic Vehicle Counter", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))
    h, w, _ = frame.shape

    # Center line y-coordinate
    line_y = h // 2

    # Model detection & tracking
    results = model.track(frame, persist=True)

    for r in results:
        if r.boxes.id is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Count only if vehicle crosses the center line
                    if track_id not in counted_ids and (line_y - offset) < cy < (line_y + offset):
                        counted_ids.add(track_id)

                    # Draw box & center
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Vehicle count and time estimation
    vehicle_count = len(counted_ids)
    vehicles_per_lane = math.ceil(vehicle_count / lane_count) if lane_count > 0 else 0
    total_time = min(vehicles_per_lane * avg_time_per_vehicle, 120)

    # Draw center line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

    # Display info
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Lanes: {lane_count}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehicles per Lane: {vehicles_per_lane}", (350, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Estimated Time: {total_time} sec", (350, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Traffic Vehicle Counter", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final Output in Terminal
print("\n=== Final Result ===")
print(f"Total Vehicles Detected: {vehicle_count}")
print(f"Lanes: {lane_count}")
print(f"Vehicles per Lane: {vehicles_per_lane}")
print(f"Estimated Green Light Time: {total_time} seconds")
