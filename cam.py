# import cv2
# from recognition_system import MissingPersonRecognitionSystem
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# MODEL_PATH = os.path.join(PROJECT_ROOT, "Cnn_pipeline","checkpoints", "face_recognition_model.pth")

# db_config = {
#     'host': 'localhost',
#     'user': 'postgres',
#     'password': '@10kechar',
#     'database': 'missing_person_db',
#     'port': 5432
# }

# system = MissingPersonRecognitionSystem(
#     model_path=MODEL_PATH,
#     db_config=db_config
# )

# cap = cv2.VideoCapture(0)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     matches = system.process_frame(
#         frame,
#         location="Main Gate Camera",
#         camera_id="CAM_01"
#     )

#     for match in matches:
#         print("🚨 MATCH FOUND:", match)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






import cv2
from recognition_system import MissingPersonRecognitionSystem
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Cnn_pipeline",
    "checkpoints",
    "face_recognition_model.pth"
)

db_config = {
    'host': 'localhost',
    'user': 'postgres',
    'password': '@10kechar',
    'database': 'missing_person_db',
    'port': 5432
}

system = MissingPersonRecognitionSystem(
    model_path=MODEL_PATH,
    db_config=db_config
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces first (for drawing all boxes)
    faces = system.recognizer.detect_face(frame)

    # Process recognition
    matches = system.process_frame(
        frame,
        location="Main Gate Camera",
        camera_id="CAM_01"
    )

    # Draw GREEN square box for all detected faces
    for face_data in faces:
        x1, y1, x2, y2 = face_data["box"]

        width = x2 - x1
        height = y2 - y1
        side = max(width, height)

        cx = x1 + width // 2
        cy = y1 + height // 2

        new_x1 = max(0, cx - side // 2)
        new_y1 = max(0, cy - side // 2)
        new_x2 = new_x1 + side
        new_y2 = new_y1 + side

        cv2.rectangle(
            frame,
            (new_x1, new_y1),
            (new_x2, new_y2),
            (0, 255, 0),  # Green
            2
        )

    # Print matched persons
    for match in matches:
        print("🚨 MATCH FOUND:", match)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
