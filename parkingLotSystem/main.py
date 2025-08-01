import cv2
import time
from plate_detection.yolo_detect import detect_plate_text, recognize_plate_id
from face_recognition.main_face_recognition import recognize_face_id
from utils import log_access

def capture_camera_frame():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)  # For /dev/video1
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def continuous_monitoring():
    """Continuous monitoring mode with no delays - processes every frame"""
    print("=== CONTINUOUS FACE RECOGNITION MODE ===")
    print("Processing every frame with no delays")
    print("Press 'q' to quit")
    print("Starting...")
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process every frame - no delays
            face_id = recognize_face_id(frame)
            plate_text = detect_plate_text(frame)
            
            # Display detection results on frame
            status_text = ""
            if face_id and plate_text:
                status_text = f"[ACCESS GRANTED] {face_id} + {plate_text}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                log_access(face_id, plate_text, access=True)
            elif face_id:
                status_text = f"[FACE DETECTED] {face_id}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif plate_text:
                status_text = f"[PLATE DETECTED] {plate_text}"
                cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "[MONITORING...]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Continuous Face Recognition', frame)
            
            # Check for quit (no delay in cv2.waitKey)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping continuous monitoring...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    frame = capture_camera_frame()
    if frame is None:
        print("Failed to capture frame.")
        return

    # Detect plate and face using the new YOLO-based systems
    plate_text = detect_plate_text(frame)
    face_id = recognize_face_id(frame)

    # Check authorization - both face and plate must be authorized
    if plate_text and face_id:
        print(f"[ACCESS GRANTED] {face_id} with vehicle {plate_text}")
        log_access(face_id, plate_text, access=True)
    elif plate_text:
        print(f"[PARTIAL ACCESS] Vehicle {plate_text} detected but no authorized face")
        log_access(None, plate_text, access=False)
    elif face_id:
        print(f"[PARTIAL ACCESS] Face {face_id} detected but no authorized vehicle")
        log_access(face_id, None, access=False)
    else:
        print("[ACCESS DENIED] No authorized face or plate detected")
        log_access(None, None, access=False)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        continuous_monitoring()
    else:
        main()
