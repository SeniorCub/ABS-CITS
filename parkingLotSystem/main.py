import cv2
from plate_detection.yolo_detect import detect_plate_text
from face_recognition.main_ace_recognition import recognize_face_id
from utils import log_access, load_known_faces, load_known_plates

def capture_camera_frame():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)  # For /dev/video1
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main():
    known_faces = load_known_faces("known_faces/")
    known_plates = load_known_plates("known_plates.txt")

    frame = capture_camera_frame()
    if frame is None:
        print("Failed to capture frame.")
        return

    plate_text = detect_plate_text(frame)
    face_id = recognize_face_id(frame, known_faces)

    if plate_text in known_plates and face_id is not None:
        print(f"[ACCESS GRANTED] {face_id} with vehicle {plate_text}")
        log_access(face_id, plate_text, access=True)
    else:
        print("[ACCESS DENIED] Unknown face or plate")
        log_access(face_id, plate_text, access=False)

if __name__ == "__main__":
    main()
