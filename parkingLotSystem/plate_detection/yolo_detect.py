import cv2
import numpy as np
import os
from datetime import datetime
import logging
import sqlite3
import time
import threading
import pandas as pd
import torch
import warnings

# Suppress hardware warnings and set CPU-only mode for compatibility
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent threading issues
torch.set_num_threads(1)  # Set single thread for CPU

# Force CPU usage to avoid hardware compatibility issues
torch.set_default_tensor_type('torch.FloatTensor')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO detection disabled.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Tesseract OCR disabled.")

from PIL import Image
import imutils

try:  
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available. EasyOCR disabled.")

try:
    import pygame
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("Warning: pygame not available. Sound notifications disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParkingLotPlateRecognition:
    def __init__(self, output_path="CapturedPlates", unknown_plates_path="unknown_plates", 
                 db_path="plate_system.db", model_path="plate_detection/model/yolo11m_car_plate_trained.pt",
                 ocr_model_path="plate_detection/model/yolo11m_car_plate_ocr.pt", 
                 authorized_plates_file="plate_detection/authorized_plates.xlsx"):
        self.output_path = output_path
        self.unknown_plates_path = unknown_plates_path
        self.db_path = db_path
        self.model_path = model_path
        self.ocr_model_path = ocr_model_path
        self.authorized_plates_file = authorized_plates_file
        
        # Load YOLO models
        self.detection_model = None
        self.ocr_model = None
        self.load_models()
        
        # Load authorized plates
        self.authorized_plates = []
        self.load_authorized_plates()

        # Plate recognition settings
        self.confidence_threshold = 0.5  # Minimum confidence for detection
        self.ocr_confidence_threshold = 0.6  # Minimum confidence for OCR
        
        # Adjusted thresholds for better recognition
        self.plate_distance_threshold = 0.6  # For matching similar plates
        self.unknown_threshold = 0.5  # Threshold for matching unknown plates

        # Capture control settings
        self.capture_interval = 10  # Capture every 10 seconds per plate
        self.last_capture_time = {}  # Track last capture time per plate
        self.last_sound_time = {}  # Track last sound notification per plate
        self.sound_interval = 5  # Play sound every 5 seconds per plate

        # Activity tracking
        self.current_detections = {}  # Currently detected plates
        self.detection_start_time = {}  # When each plate was first detected
        self.min_detection_duration = 2  # Minimum seconds before logging visit

        # Create directories if they don't exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.unknown_plates_path, exist_ok=True)

        # Initialize database
        self.init_database()

        # Initialize sound system
        self.init_sound_system()

        # Initialize EasyOCR reader if available
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)  # Force CPU usage
                logger.info("EasyOCR reader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                # Set instance variable to track EasyOCR availability for this instance
                self.easyocr_available = False
        else:
            self.easyocr_available = False

    def load_models(self):
        """Load YOLO models for plate detection and OCR with hardware compatibility checks"""
        try:
            # Force CPU usage and disable hardware acceleration to avoid compatibility issues
            if YOLO_AVAILABLE:
                # Set device to CPU explicitly
                device = 'cpu'
                
                if os.path.exists(self.model_path):
                    self.detection_model = YOLO(self.model_path)
                    # Force CPU usage and disable problematic optimizations
                    if hasattr(self.detection_model, 'device'):
                        self.detection_model.to(device)
                    logger.info(f"Loaded plate detection model on {device}: {self.model_path}")
                else:
                    logger.warning(f"Detection model not found: {self.model_path}")
                    
                if os.path.exists(self.ocr_model_path):
                    self.ocr_model = YOLO(self.ocr_model_path)
                    # Force CPU usage and disable problematic optimizations
                    if hasattr(self.ocr_model, 'device'):
                        self.ocr_model.to(device)
                    logger.info(f"Loaded OCR model on {device}: {self.ocr_model_path}")
                else:
                    logger.warning(f"OCR model not found: {self.ocr_model_path}")
            else:
                logger.warning("YOLO not available, using traditional detection methods only")
                
        except Exception as e:
            logger.error(f"Error loading YOLO models: {e}")
            logger.info("Falling back to traditional detection methods")
    
    def load_authorized_plates(self):
        """Load authorized plates from Excel file"""
        try:
            if os.path.exists(self.authorized_plates_file):
                df = pd.read_excel(self.authorized_plates_file)
                # Assume the first column contains plate numbers
                self.authorized_plates = df.iloc[:, 0].astype(str).str.upper().tolist()
                logger.info(f"Loaded {len(self.authorized_plates)} authorized plates")
            else:
                logger.warning(f"Authorized plates file not found: {self.authorized_plates_file}")
                self.authorized_plates = []
        except Exception as e:
            logger.error(f"Error loading authorized plates: {e}")
            self.authorized_plates = []
    
    def init_sound_system(self):
        """Initialize pygame for sound notifications"""
        global SOUND_AVAILABLE
        if SOUND_AVAILABLE:
            try:
                pygame.mixer.init()
                # Load sound files
                self.known_plate_sound = None
                self.unknown_plate_sound = None
                
                if os.path.exists("known_plate_beep.wav"):
                    self.known_plate_sound = pygame.mixer.Sound("known_plate_beep.wav")
                if os.path.exists("unknown_plate_beep.wav"):
                    self.unknown_plate_sound = pygame.mixer.Sound("unknown_plate_beep.wav")
                    
                logger.info("Sound system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize sound system: {e}")
                SOUND_AVAILABLE = False

    def play_sound(self, is_authorized):
        """Play appropriate sound for plate recognition"""
        if not SOUND_AVAILABLE:
            return
            
        try:
            if is_authorized and self.known_plate_sound:
                self.known_plate_sound.play()
            elif not is_authorized and self.unknown_plate_sound:
                self.unknown_plate_sound.play()
        except Exception as e:
            logger.error(f"Error playing sound: {e}")

    def init_database(self):
        """Initialize SQLite database for plate recognition"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create authorized_plates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS authorized_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    owner_name TEXT,
                    vehicle_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Create unknown_plates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    visit_count INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Create plate_activity_log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plate_activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    is_authorized BOOLEAN NOT NULL,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    detection_method TEXT DEFAULT 'YOLO'
                )
            ''')

            # Create plate_visits table (similar to face recognition visits)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plate_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds INTEGER,
                    is_authorized BOOLEAN,
                    max_confidence REAL
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def detect_plates_yolo(self, frame):
        """Detect license plates using YOLO model with hardware compatibility"""
        try:
            if not YOLO_AVAILABLE or self.detection_model is None:
                logger.warning("Detection model not loaded, falling back to traditional method")
                return self.detect_plates_traditional(frame)
            
            # Run YOLO detection with CPU-only mode and error handling
            with torch.no_grad():  # Disable gradient computation for inference
                results = self.detection_model(frame, conf=self.confidence_threshold, device='cpu', verbose=False)
            
            detected_plates = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates with safe tensor handling
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            confidence = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                        except Exception:
                            # Fallback for different tensor formats
                            coords = box.xyxy[0]
                            x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                            confidence = float(box.conf[0])
                        
                        if confidence >= self.confidence_threshold:
                            # Extract plate region
                            plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
                            
                            # Recognize text from plate
                            plate_text, ocr_confidence = self.recognize_plate_text_yolo(plate_region)
                            
                            if plate_text and ocr_confidence >= self.ocr_confidence_threshold:
                                detected_plates.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': float(confidence),
                                    'plate_text': plate_text,
                                    'ocr_confidence': float(ocr_confidence),
                                    'region': plate_region
                                })
            
            return detected_plates
            
        except Exception as e:
            logger.error(f"Error in YOLO plate detection: {e}")
            return []

    def recognize_plate_text_yolo(self, plate_image):
        """Recognize text from plate using YOLO OCR model with hardware compatibility"""
        try:
            if not YOLO_AVAILABLE or self.ocr_model is None:
                logger.warning("OCR model not loaded, falling back to traditional OCR")
                return self.recognize_plate_text_traditional(plate_image)
            
            # Use YOLO OCR model with CPU-only mode and error handling
            with torch.no_grad():  # Disable gradient computation for inference
                results = self.ocr_model(plate_image, conf=self.ocr_confidence_threshold, device='cpu', verbose=False)
            
            detected_text = ""
            total_confidence = 0
            text_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        try:
                            confidence = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
                        except Exception:
                            confidence = float(box.conf[0])
                            
                        if confidence >= self.ocr_confidence_threshold:
                            # Get the class name (which should be the character/text)
                            try:
                                class_id = int(box.cls[0].cpu().numpy() if hasattr(box.cls[0], 'cpu') else box.cls[0])
                            except Exception:
                                class_id = int(box.cls[0])
                                
                            if hasattr(result, 'names') and class_id in result.names:
                                char = result.names[class_id]
                                detected_text += char
                                total_confidence += confidence
                                text_count += 1
            
            if text_count > 0:
                avg_confidence = total_confidence / text_count
                return detected_text.upper(), avg_confidence * 100
            else:
                # Fallback to traditional OCR
                return self.recognize_plate_text_traditional(plate_image)
                
        except Exception as e:
            logger.error(f"Error in YOLO OCR: {e}")
            return self.recognize_plate_text_traditional(plate_image)

    def detect_plates_traditional(self, frame):
        """Traditional plate detection method (fallback)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 170, 200)
            
            # Find contours
            contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            detected_plates = []
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    area = cv2.contourArea(contour)

                    if (2 <= aspect_ratio <= 5 and 2000 <= area <= 20000):
                        plate_region = frame[y:y + h, x:x + w]
                        plate_text, confidence = self.recognize_plate_text_traditional(plate_region)
                        
                        if plate_text and confidence >= 60:
                            detected_plates.append({
                                'bbox': (x, y, x + w, y + h),
                                'confidence': 0.8,  # Default confidence for traditional method
                                'plate_text': plate_text,
                                'ocr_confidence': confidence,
                                'region': plate_region
                            })

            return detected_plates

        except Exception as e:
            logger.error(f"Error in traditional plate detection: {e}")
            return []

    def recognize_plate_text_traditional(self, plate_image):
        """Traditional OCR method using available OCR engines"""
        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Try EasyOCR first if available
            if EASYOCR_AVAILABLE and hasattr(self, 'reader') and self.reader is not None:
                try:
                    results = self.reader.readtext(thresh)
                    if results:
                        plate_text = " ".join([result[1] for result in results])
                        avg_confidence = sum([result[2] for result in results]) / len(results) * 100
                        return plate_text.strip(), avg_confidence
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            # Fallback to Tesseract if available
            if TESSERACT_AVAILABLE:
                try:
                    # Preprocess image for better OCR
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    thresh = cv2.medianBlur(thresh, 3)
                    
                    # Configure Tesseract for license plates
                    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(thresh, config=custom_config)
                    
                    if text.strip():
                        return text.strip(), 75.0  # Default confidence for Tesseract
                except Exception as e:
                    logger.warning(f"Tesseract OCR failed: {e}")
            
            # If both OCR methods fail, return empty result
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Error in traditional OCR: {e}")
            return "", 0.0

    def is_plate_authorized(self, plate_text):
        """Check if a plate is in the authorized list"""
        clean_plate = plate_text.replace(" ", "").upper()
        for auth_plate in self.authorized_plates:
            if clean_plate == auth_plate.replace(" ", "").upper():
                return True
        return False

    def process_frame(self, frame):
        """Process a single frame for plate detection and recognition"""
        try:
            # Detect plates using YOLO (with fallback to traditional method)
            detected_plates = self.detect_plates_yolo(frame)
            
            results = []
            for plate_data in detected_plates:
                plate_text = plate_data['plate_text']
                confidence = plate_data['confidence']
                ocr_confidence = plate_data['ocr_confidence']
                bbox = plate_data['bbox']
                
                # Check if plate is authorized
                is_authorized = self.is_plate_authorized(plate_text)
                
                # Update tracking and database
                self.update_plate_tracking(plate_text, is_authorized, confidence)
                
                # Play sound notification
                if self.should_play_sound(plate_text):
                    self.play_sound(is_authorized)
                    self.last_sound_time[plate_text] = time.time()
                
                # Save capture if needed
                if self.should_capture_plate(plate_text):
                    image_path = self.save_plate_capture(frame, plate_data)
                    self.last_capture_time[plate_text] = time.time()
                else:
                    image_path = None
                
                # Log activity
                self.log_plate_activity(plate_text, is_authorized, confidence, image_path)
                
                results.append({
                    'plate_text': plate_text,
                    'is_authorized': is_authorized,
                    'confidence': confidence,
                    'ocr_confidence': ocr_confidence,
                    'bbox': bbox,
                    'image_path': image_path
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []

    def should_capture_plate(self, plate_text):
        """Check if enough time has passed to capture this plate again"""
        current_time = time.time()
        last_time = self.last_capture_time.get(plate_text, 0)
        return (current_time - last_time) >= self.capture_interval

    def should_play_sound(self, plate_text):
        """Check if enough time has passed to play sound for this plate"""
        current_time = time.time()
        last_time = self.last_sound_time.get(plate_text, 0)
        return (current_time - last_time) >= self.sound_interval

    def save_plate_capture(self, frame, plate_data):
        """Save captured plate image with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_text = plate_data['plate_text'].replace(" ", "_")
            confidence = plate_data['confidence']
            is_authorized = self.is_plate_authorized(plate_data['plate_text'])
            
            # Choose directory based on authorization status
            save_dir = self.output_path if is_authorized else self.unknown_plates_path
            
            filename = f"{timestamp}_{plate_text}_conf_{confidence:.1f}.jpg"
            filepath = os.path.join(save_dir, filename)

            # Save the plate region
            cv2.imwrite(filepath, plate_data['region'])

            logger.info(f"Saved plate capture: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save plate capture: {e}")
            return None

    def update_plate_tracking(self, plate_text, is_authorized, confidence):
        """Update plate tracking information in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if is_authorized:
                # Update or insert in authorized_plates table
                cursor.execute('''
                    INSERT OR IGNORE INTO authorized_plates (plate_number, created_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                ''', (plate_text,))
            else:
                # Update or insert in unknown_plates table
                cursor.execute('''
                    INSERT INTO unknown_plates (plate_number, first_seen, last_seen, visit_count)
                    VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT(plate_number) DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP,
                    visit_count = visit_count + 1
                ''', (plate_text,))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update plate tracking: {e}")

    def log_plate_activity(self, plate_text, is_authorized, confidence, image_path=None):
        """Log plate detection activity to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO plate_activity_log (plate_number, is_authorized, confidence, image_path)
                VALUES (?, ?, ?, ?)
            ''', (plate_text, is_authorized, confidence, image_path))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log plate activity: {e}")

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            bbox = detection['bbox']
            plate_text = detection['plate_text']
            is_authorized = detection['is_authorized']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on authorization
            color = (0, 255, 0) if is_authorized else (0, 0, 255)  # Green for authorized, Red for unauthorized
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{plate_text} ({'AUTH' if is_authorized else 'UNAUTH'}) {confidence:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def run_camera_detection(self, camera_index=1):
        """Run live camera detection similar to face recognition system"""
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                logger.error(f"Cannot open camera {camera_index}")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            print("\n=== Parking Lot Plate Recognition System ===")
            print("Controls:")
            print("- Press 'q' to quit")
            print("- Press 's' to manually save current frame")
            print("- Press 'l' to list recognized plates")
            print("- Press 'r' to reload authorized plates")
            logger.info(f"Starting plate detection on camera {camera_index}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Process frame for plate detection
                detections = self.process_frame(frame)
                
                # Draw detections on frame
                frame = self.draw_detections(frame, detections)
                
                # Display frame
                cv2.imshow('Parking Lot Plate Recognition', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"manual_capture_{timestamp}.jpg", frame)
                    print(f"Frame saved as manual_capture_{timestamp}.jpg")
                elif key == ord('l'):
                    self.list_recent_plates()
                elif key == ord('r'):
                    self.load_authorized_plates()
                    print("Authorized plates reloaded")

        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Error in camera detection: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def list_recent_plates(self, hours=24):
        """List recently detected plates"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT plate_number, is_authorized, confidence, timestamp
                FROM plate_activity_log
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
                LIMIT 20
            '''.format(hours))
            
            results = cursor.fetchall()
            
            print(f"\nRecent plates (last {hours} hours):")
            print("-" * 60)
            for plate, is_auth, conf, timestamp in results:
                auth_status = "AUTHORIZED" if is_auth else "UNAUTHORIZED"
                print(f"{timestamp} | {plate:15} | {auth_status:12} | {conf:.1f}%")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error listing recent plates: {e}")


# Global instance for use by main.py
_plate_recognition_system = None

def get_plate_recognition_system():
    """Get or create the global plate recognition system"""
    global _plate_recognition_system
    if _plate_recognition_system is None:
        _plate_recognition_system = ParkingLotPlateRecognition()
    return _plate_recognition_system

def detect_plate_text(frame):
    """Main function to be called by main.py - returns detected plate text"""
    try:
        system = get_plate_recognition_system()
        detections = system.process_frame(frame)
        
        # Return the first authorized plate found, or None if no plates detected
        for detection in detections:
            if detection['is_authorized']:
                return detection['plate_text']
        
        # If no authorized plates, return the first detected plate
        if detections:
            return detections[0]['plate_text']
            
        return None
        
    except Exception as e:
        logger.error(f"Error in detect_plate_text: {e}")
        return None

def recognize_plate_id(frame, known_plates=None):
    """Alternative function similar to face recognition's recognize_face_id"""
    try:
        system = get_plate_recognition_system()
        detections = system.process_frame(frame)
        
        for detection in detections:
            plate_text = detection['plate_text']
            if detection['is_authorized']:
                return plate_text
        
        return None
        
    except Exception as e:
        logger.error(f"Error in recognize_plate_id: {e}")
        return None


if __name__ == "__main__":
    """Main function with command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Parking Lot Plate Recognition System')
    parser.add_argument('--mode', choices=['camera', 'video', 'report', 'list'],
                        default='camera', help='Operation mode')
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--days', type=int, default=7, help='Number of days for report')

    args = parser.parse_args()

    # Initialize the system
    plate_system = ParkingLotPlateRecognition()

    try:
        if args.mode == 'camera':
            plate_system.run_camera_detection(camera_index=args.camera)
        elif args.mode == 'report':
            plate_system.generate_plate_report(days=args.days)
        elif args.mode == 'list':
            plate_system.list_recent_plates()

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info("Program ended")