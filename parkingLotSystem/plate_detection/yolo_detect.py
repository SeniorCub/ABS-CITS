import cv2
import numpy as np
import os
from datetime import datetime
import logging
import sqlite3
import time
import threading
import pytesseract
from PIL import Image
import imutils
import easyocr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParkingLotPlateRecognition:
    def __init__(self, output_path="CapturedPlates", db_path="parking_system.db"):
        self.output_path = output_path
        self.db_path = db_path

        # Plate recognition settings
        self.min_plate_area = 2000  # Minimum area for plate detection
        self.max_plate_area = 20000  # Maximum area for plate detection
        self.plate_aspect_ratio_range = (2, 5)  # Width/height ratio range for plates

        # Capture control settings
        self.capture_interval = 30  # Capture every 30 seconds per plate
        self.last_capture_time = {}  # Track last capture time per plate

        # Create directories if they don't exist
        os.makedirs(self.output_path, exist_ok=True)

        # Initialize database
        self.init_database()

        # Initialize OCR reader
        self.reader = easyocr.Reader(['en'])  # English only for now

    def init_database(self):
        """Initialize SQLite database for plate recognition"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create plates table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    visit_count INTEGER DEFAULT 1,
                    is_authorized BOOLEAN DEFAULT 0,
                    owner_name TEXT,
                    additional_info TEXT
                )
            ''')

            # Create plate_activity_log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plate_activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    plate_number TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL,
                    image_path TEXT
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("Plate database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing plate database: {str(e)}")

    def log_plate_activity(self, plate_number, action, confidence=None, image_path=None):
        """Log plate activity to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO plate_activity_log (plate_number, action, confidence, image_path)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, action, confidence, image_path))

            conn.commit()
            conn.close()
            logger.info(f"Plate activity logged: {plate_number} - {action}")

        except Exception as e:
            logger.error(f"Failed to log plate activity: {e}")

    def update_plate_in_db(self, plate_number):
        """Update plate information in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if plate exists
            cursor.execute('SELECT id FROM plates WHERE plate_number = ?', (plate_number,))
            existing = cursor.fetchone()

            if existing:
                # Update existing plate
                cursor.execute('''
                    UPDATE plates 
                    SET last_seen = CURRENT_TIMESTAMP, 
                        visit_count = visit_count + 1
                    WHERE plate_number = ?
                ''', (plate_number,))
            else:
                # Insert new plate
                cursor.execute('''
                    INSERT INTO plates (plate_number, first_seen, last_seen, visit_count)
                    VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
                ''', (plate_number,))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update plate in database: {e}")

    def detect_plates(self, frame):
        """Detect license plates in the frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter to reduce noise while keeping edges sharp
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # Find edges
            edged = cv2.Canny(gray, 170, 200)

            # Find contours
            contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            # Sort contours by area (descending) and keep top 10
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            plate_contours = []

            for contour in contours:
                # Approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Check if contour is rectangular (4 sides)
                if len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)

                    # Check if aspect ratio is within expected range for plates
                    if (self.plate_aspect_ratio_range[0] <= aspect_ratio <= self.plate_aspect_ratio_range[1] and
                            self.min_plate_area <= cv2.contourArea(contour) <= self.max_plate_area):
                        plate_contours.append(approx)

            return plate_contours

        except Exception as e:
            logger.error(f"Error detecting plates: {str(e)}")
            return []

    def recognize_plate_text(self, plate_image):
        """Recognize text from a plate image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Use EasyOCR for text recognition
            results = self.reader.readtext(thresh)

            if results:
                # Combine all detected text
                plate_text = " ".join([result[1] for result in results])
                confidence = np.mean([result[2] for result in results]) if results else 0
                return plate_text.upper(), confidence * 100

            return None, 0

        except Exception as e:
            logger.error(f"Error recognizing plate text: {str(e)}")
            return None, 0

    def draw_plate_boxes(self, frame, plates):
        """Draw bounding boxes around detected plates"""
        for plate in plates:
            # Get bounding rectangle coordinates
            x, y, w, h = cv2.boundingRect(plate)

            # Draw rectangle around plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put plate label
            cv2.putText(frame, "License Plate", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def save_plate_capture(self, frame, plate_contour, plate_text, confidence):
        """Save plate image with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Get plate region
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate_region = frame[y:y + h, x:x + w]

            # Create filename
            filename = f"{timestamp}_{plate_text}_{confidence:.1f}.jpg"
            filepath = os.path.join(self.output_path, filename)

            # Save plate image
            cv2.imwrite(filepath, plate_region)

            # Log to database
            self.log_plate_activity(plate_text, "detected", confidence, filepath)
            self.update_plate_in_db(plate_text)

            logger.info(f"Saved plate capture: {filename}")

            return filepath

        except Exception as e:
            logger.error(f"Error saving plate capture: {str(e)}")
            return None

    def should_capture_plate(self, plate_text):
        """Check if enough time has passed to capture this plate again"""
        current_time = time.time()
        if plate_text not in self.last_capture_time:
            return True
        return current_time - self.last_capture_time[plate_text] >= self.capture_interval

    def run_camera_detection(self, camera_index=0):
        """Run real-time plate detection from camera"""
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
            logger.info(f"Starting plate detection on camera {camera_index}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Detect plates
                plate_contours = self.detect_plates(frame)

                # Process each detected plate
                for contour in plate_contours:
                    # Get plate region
                    x, y, w, h = cv2.boundingRect(contour)
                    plate_region = frame[y:y + h, x:x + w]

                    # Recognize plate text
                    plate_text, confidence = self.recognize_plate_text(plate_region)

                    if plate_text and confidence > 60:  # Minimum confidence threshold
                        # Draw plate info on frame
                        cv2.putText(frame, f"{plate_text} ({confidence:.1f}%)",
                                    (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

                        # Save plate if enough time has passed
                        if self.should_capture_plate(plate_text):
                            self.save_plate_capture(frame, contour, plate_text, confidence)
                            self.last_capture_time[plate_text] = time.time()

                # Draw plate contours on frame
                frame = self.draw_plate_boxes(frame, plate_contours)

                # Display frame
                cv2.imshow('Parking Lot Plate Recognition', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_capture_{timestamp}.jpg"
                    filepath = os.path.join(self.output_path, filename)
                    cv2.imwrite(filepath, frame)
                    logger.info(f"Manual capture saved: {filename}")
                elif key == ord('l'):
                    self.list_recognized_plates()

            cap.release()
            cv2.destroyAllWindows()
            logger.info("Plate detection stopped")

        except Exception as e:
            logger.error(f"Error in plate detection: {str(e)}")
        finally:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass

    def list_recognized_plates(self):
        """List all recognized plates with their information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT plate_number, first_seen, last_seen, visit_count, is_authorized, owner_name
                FROM plates 
                ORDER BY last_seen DESC
                LIMIT 50
            ''')

            rows = cursor.fetchall()
            conn.close()

            print("\n=== Recognized Plates ===")
            if not rows:
                print("No plates in database")
                return

            for row in rows:
                plate_number, first_seen, last_seen, visit_count, is_authorized, owner_name = row
                status = "Authorized" if is_authorized else "Unauthorized"
                owner = owner_name if owner_name else "Unknown"

                print(f"Plate: {plate_number}")
                print(f"  Status: {status}")
                print(f"  Owner: {owner}")
                print(f"  Visits: {visit_count}")
                print(f"  First Seen: {first_seen}")
                print(f"  Last Seen: {last_seen}")
                print("-" * 40)

        except Exception as e:
            logger.error(f"Error listing recognized plates: {str(e)}")

    def process_video_file(self, video_path):
        """Process a video file for plate recognition"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Processing video: {video_path}")
            logger.info(f"Total frames: {total_frames}, FPS: {fps}")

            frame_count = 0
            process_every_n_frames = int(fps / 2)  # Process 2 frames per second

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process every N frames
                if frame_count % process_every_n_frames == 0:
                    plate_contours = self.detect_plates(frame)

                    for contour in plate_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        plate_region = frame[y:y + h, x:x + w]

                        plate_text, confidence = self.recognize_plate_text(plate_region)

                        if plate_text and confidence > 60:
                            self.save_plate_capture(frame, contour, plate_text, confidence)

                            # Display progress
                            progress = (frame_count / total_frames) * 100
                            logger.info(f"Progress: {progress:.1f}% - Found plate: {plate_text}")

                # Display frame (optional)
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video processing completed")

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")

    def generate_plate_report(self, days=7):
        """Generate a report of plate recognition activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            print(f"\n=== Plate Recognition Report (Last {days} days) ===")
            print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            # Total plate activity
            cursor.execute('''
                SELECT COUNT(*) as total_detections
                FROM plate_activity_log 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days))

            total_detections = cursor.fetchone()[0]

            print(f"\nTotal Plate Detections: {total_detections}")

            # Unique plates detected
            cursor.execute('''
                SELECT COUNT(DISTINCT plate_number) as unique_plates
                FROM plate_activity_log 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days))

            unique_plates = cursor.fetchone()[0]

            print(f"Unique Plates Detected: {unique_plates}")

            # Most frequent plates
            cursor.execute('''
                SELECT plate_number, COUNT(*) as detections
                FROM plate_activity_log 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY plate_number
                ORDER BY detections DESC
                LIMIT 10
            '''.format(days))

            frequent_plates = cursor.fetchall()

            print("\nTop 10 Most Frequent Plates:")
            print("-" * 40)
            for plate, count in frequent_plates:
                print(f"{plate}: {count} detections")

            # Authorized vs unauthorized
            cursor.execute('''
                SELECT p.is_authorized, COUNT(*) as count
                FROM plate_activity_log a
                JOIN plates p ON a.plate_number = p.plate_number
                WHERE a.timestamp >= datetime('now', '-{} days')
                GROUP BY p.is_authorized
            '''.format(days))

            auth_stats = cursor.fetchall()

            print("\nAuthorization Status:")
            print("-" * 40)
            for is_auth, count in auth_stats:
                status = "Authorized" if is_auth else "Unauthorized"
                print(f"{status}: {count} detections")

            conn.close()

        except Exception as e:
            logger.error(f"Error generating plate report: {str(e)}")


def main():
    """Main function with command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Parking Lot Plate Recognition System')
    parser.add_argument('--mode', choices=['camera', 'video', 'report', 'list'],
                        default='camera', help='Operation mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--days', type=int, default=7, help='Number of days for report')

    args = parser.parse_args()

    # Initialize the system
    plate_system = ParkingLotPlateRecognition()

    try:
        if args.mode == 'camera':
            plate_system.run_camera_detection(camera_index=args.camera)
        elif args.mode == 'video':
            if not args.video:
                print("Error: --video path is required for video mode")
                return
            plate_system.process_video_file(args.video)
        elif args.mode == 'report':
            plate_system.generate_plate_report(days=args.days)
        elif args.mode == 'list':
            plate_system.list_recognized_plates()

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info("Program ended")


if __name__ == "__main__":
    main()