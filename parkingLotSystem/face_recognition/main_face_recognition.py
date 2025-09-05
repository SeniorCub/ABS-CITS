import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import logging
import sqlite3
import pickle
import time
import threading
import base64

try:
    import pygame

    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("Warning: pygame not available. Sound notifications disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParkingLotFaceRecognition:
    def __init__(self, known_faces_path="People", output_path="Captured",
                 unknown_faces_path="unknown_faces", db_path="parking_system.db"):
        self.known_faces_path = known_faces_path
        self.output_path = output_path
        self.unknown_faces_path = unknown_faces_path
        self.db_path = db_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.unknown_face_encodings = []
        self.unknown_face_data = []

        # Adjusted thresholds for better recognition
        self.face_distance_threshold = 0.6  # Increased threshold for known faces
        self.unknown_threshold = 0.5  # Threshold for matching unknown faces

        # Capture control settings - Set to 0 for constant detection
        self.capture_interval = 0  # Capture continuously (no delay)
        self.last_capture_time = {}  # Track last capture time per person
        self.last_sound_time = {}  # Track last sound notification per person
        self.sound_interval = 0  # Play sound continuously (no delay)

        # Activity tracking
        self.current_detections = {}  # Currently detected faces
        self.detection_start_time = {}  # When each person was first detected
        self.min_detection_duration = 0  # No minimum detection time (instant logging)

        # Create directories if they don't exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.unknown_faces_path, exist_ok=True)

        # Initialize database
        self.init_database()

        # Initialize sound system
        self.init_sound_system()

        # Load known faces and unknown faces database
        self.load_known_faces()
        self.load_unknown_faces_from_db()

    def init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create known_faces table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create unknown_faces table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encoding BLOB NOT NULL,
                    assigned_name TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    visit_count INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Create activity_log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    person_name TEXT NOT NULL,
                    face_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    additional_info TEXT,
                    confidence REAL
                )
            ''')

            # Create captures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    person_name TEXT NOT NULL,
                    face_type TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    confidence REAL
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    def init_sound_system(self):
        """Initialize pygame sound system"""
        if SOUND_AVAILABLE:
            try:
                pygame.mixer.init()
                # Create simple beep sounds programmatically
                self.create_sound_files()
                self.known_sound = pygame.mixer.Sound("known_beep.wav")
                self.unknown_sound = pygame.mixer.Sound("unknown_beep.wav")
                logger.info("Sound system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sound system: {e}")
                self.known_sound = None
                self.unknown_sound = None
        else:
            self.known_sound = None
            self.unknown_sound = None

    def create_sound_files(self):
        """Create simple beep sound files"""
        try:
            import numpy as np
            import wave

            # Create known person sound (higher pitch, double beep)
            sample_rate = 22050
            duration = 0.2
            frequency1 = 700
            frequency2 = 1000

            # Generate double beep for known person
            t = np.linspace(0, duration, int(sample_rate * duration))
            beep1 = np.sin(2 * np.pi * frequency1 * t) * 0.3
            silence = np.zeros(int(sample_rate * 0.1))
            beep2 = np.sin(2 * np.pi * frequency2 * t) * 0.3
            known_wave = np.concatenate([beep1, silence, beep2])

            # Save known person sound
            with wave.open("known_beep.wav", 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((known_wave * 32767).astype(np.int16).tobytes())

            # Create unknown person sound (lower pitch, single beep)
            unknown_wave = np.sin(2 * np.pi * 600 * t) * 0.3

            # Save unknown person sound
            with wave.open("unknown_beep.wav", 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((unknown_wave * 32767).astype(np.int16).tobytes())

        except Exception as e:
            logger.warning(f"Failed to create sound files: {e}")

    def play_sound(self, sound_type):
        """Play sound notification"""
        if not SOUND_AVAILABLE:
            return

        try:
            if sound_type == "known" and self.known_sound:
                threading.Thread(target=self.known_sound.play, daemon=True).start()
            elif sound_type == "unknown" and self.unknown_sound:
                threading.Thread(target=self.unknown_sound.play, daemon=True).start()
        except Exception as e:
            logger.warning(f"Failed to play sound: {e}")

    def log_activity(self, person_name, face_type, action, confidence=None, additional_info=None):
        """Log activity to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO activity_log (person_name, face_type, action, confidence, additional_info)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_name, face_type, action, confidence, additional_info))

            conn.commit()
            conn.close()
            logger.info(f"Activity logged: {person_name} - {action}")

        except Exception as e:
            logger.error(f"Failed to log activity: {e}")

    def save_known_face_to_db(self, name, encoding):
        """Save known face encoding to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serialize encoding
            encoding_blob = pickle.dumps(encoding)

            cursor.execute('''
                INSERT OR REPLACE INTO known_faces (name, encoding, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (name, encoding_blob))

            conn.commit()
            conn.close()
            logger.info(f"Known face saved to database: {name}")

        except Exception as e:
            logger.error(f"Failed to save known face to database: {e}")

    def load_known_faces(self):
        """Load known faces from files and database"""
        try:
            # First, load from database
            self.load_known_faces_from_db()

            # Then, load from files and add any new ones to database
            if not os.path.exists(self.known_faces_path):
                logger.warning(f"Known faces directory '{self.known_faces_path}' does not exist, creating it...")
                os.makedirs(self.known_faces_path, exist_ok=True)
                return

            face_files = [f for f in os.listdir(self.known_faces_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not face_files:
                logger.warning(f"No image files found in '{self.known_faces_path}'")
                return

            logger.info(f"Loading {len(face_files)} known faces from files...")

            for filename in face_files:
                filepath = os.path.join(self.known_faces_path, filename)
                try:
                    # Load image
                    image = cv2.imread(filepath)
                    if image is None:
                        logger.warning(f"Could not load image: {filepath}")
                        continue

                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_image)

                    if len(face_encodings) == 0:
                        logger.warning(f"No face found in image: {filename}")
                        continue

                    if len(face_encodings) > 1:
                        logger.warning(f"Multiple faces found in {filename}, using the first one")

                    # Store the first face encoding and name
                    name = os.path.splitext(filename)[0]

                    # Check if this name already exists in our loaded faces
                    if name not in self.known_face_names:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        # Save to database
                        self.save_known_face_to_db(name, face_encodings[0])
                        logger.info(f"Loaded and saved known face: {name}")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

            logger.info(f"Successfully loaded {len(self.known_face_encodings)} known faces")

        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")

    def load_known_faces_from_db(self):
        """Load known faces from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT name, encoding FROM known_faces ORDER BY name')
            rows = cursor.fetchall()

            self.known_face_encodings = []
            self.known_face_names = []

            for name, encoding_blob in rows:
                encoding = pickle.loads(encoding_blob)
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

            conn.close()
            logger.info(f"Loaded {len(self.known_face_encodings)} known faces from database")

        except Exception as e:
            logger.error(f"Error loading known faces from database: {str(e)}")

    def save_unknown_face_to_db(self, encoding):
        """Save unknown face encoding to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serialize encoding
            encoding_blob = pickle.dumps(encoding)

            cursor.execute('''
                INSERT INTO unknown_faces (encoding, first_seen, last_seen, visit_count)
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            ''', (encoding_blob,))

            unknown_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(f"Unknown face saved to database with ID: {unknown_id}")
            return unknown_id

        except Exception as e:
            logger.error(f"Failed to save unknown face to database: {e}")
            return None

    def update_unknown_face_visit(self, unknown_id):
        """Update unknown face visit count and last seen time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE unknown_faces 
                SET last_seen = CURRENT_TIMESTAMP, visit_count = visit_count + 1
                WHERE id = ?
            ''', (unknown_id,))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update unknown face visit: {e}")

    def load_unknown_faces_from_db(self):
        """Load unknown faces from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, encoding, assigned_name, first_seen, last_seen, visit_count 
                FROM unknown_faces 
                WHERE is_active = 1 
                ORDER BY first_seen
            ''')
            rows = cursor.fetchall()

            self.unknown_face_encodings = []
            self.unknown_face_data = []

            for row in rows:
                unknown_id, encoding_blob, assigned_name, first_seen, last_seen, visit_count = row
                encoding = pickle.loads(encoding_blob)

                self.unknown_face_encodings.append(encoding)
                self.unknown_face_data.append({
                    'id': unknown_id,
                    'assigned_name': assigned_name,
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'visit_count': visit_count
                })

            conn.close()
            logger.info(f"Loaded {len(self.unknown_face_encodings)} unknown faces from database")

        except Exception as e:
            logger.error(f"Error loading unknown faces from database: {str(e)}")

    def should_capture(self, person_id):
        """Check if enough time has passed to capture again for this person"""
        current_time = time.time()
        if person_id not in self.last_capture_time:
            return True
        return current_time - self.last_capture_time[person_id] >= self.capture_interval

    def should_play_sound(self, person_id):
        """Check if enough time has passed to play sound again for this person"""
        current_time = time.time()
        if person_id not in self.last_sound_time:
            return True
        return current_time - self.last_sound_time[person_id] >= self.sound_interval

    def recognize_face(self, frame):
        """Recognize faces in the given frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if not face_locations:
                return []

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized_faces = []
            current_time = time.time()

            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Unknown"
                confidence = 0
                face_type = "unknown"
                unknown_id = None
                person_id = None

                # First check against known faces
                if len(self.known_face_encodings) > 0:
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )

                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]

                    logger.debug(f"Best match distance: {min_distance:.3f}, threshold: {self.face_distance_threshold}")

                    if min_distance < self.face_distance_threshold:
                        name = self.known_face_names[best_match_index]
                        confidence = max(0, (1 - min_distance) * 100)
                        face_type = "known"
                        person_id = f"known_{name}"

                        # Track detection
                        if person_id not in self.current_detections:
                            self.current_detections[person_id] = True
                            self.detection_start_time[person_id] = current_time
                            self.log_activity(name, "known", "detected", confidence)

                        # Play sound if enough time has passed
                        if self.should_play_sound(person_id):
                            self.play_sound("known")
                            self.last_sound_time[person_id] = current_time

                        logger.info(f"Known person recognized: {name} (confidence: {confidence:.1f}%)")

                # If not found in known faces, check unknown faces
                if face_type == "unknown" and len(self.unknown_face_encodings) > 0:
                    unknown_distances = face_recognition.face_distance(
                        self.unknown_face_encodings, face_encoding
                    )

                    best_unknown_index = np.argmin(unknown_distances)
                    min_unknown_distance = unknown_distances[best_unknown_index]

                    logger.debug(
                        f"Best unknown match distance: {min_unknown_distance:.3f}, threshold: {self.unknown_threshold}")

                    if min_unknown_distance < self.unknown_threshold:
                        unknown_data = self.unknown_face_data[best_unknown_index]
                        unknown_id = unknown_data['id']
                        assigned_name = unknown_data['assigned_name']

                        if assigned_name:
                            name = assigned_name
                        else:
                            name = f"Unknown_{unknown_id}"

                        confidence = max(0, (1 - min_unknown_distance) * 100)
                        face_type = "repeat_unknown"
                        person_id = f"unknown_{unknown_id}"

                        # Update visit count in database
                        self.update_unknown_face_visit(unknown_id)

                        # Update local data
                        self.unknown_face_data[best_unknown_index]['visit_count'] += 1

                        # Track detection
                        if person_id not in self.current_detections:
                            self.current_detections[person_id] = True
                            self.detection_start_time[person_id] = current_time
                            self.log_activity(name, "repeat_unknown", "detected", confidence)

                        # Play sound if enough time has passed
                        if self.should_play_sound(person_id):
                            self.play_sound("unknown")
                            self.last_sound_time[person_id] = current_time

                        logger.info(f"Repeat unknown person: {name} (confidence: {confidence:.1f}%)")

                # If completely new unknown face
                if face_type == "unknown":
                    unknown_id = self.save_unknown_face_to_db(face_encoding)
                    if unknown_id is not None:
                        name = f"Unknown_{unknown_id}"
                        person_id = f"unknown_{unknown_id}"

                        # Add to local arrays
                        self.unknown_face_encodings.append(face_encoding)
                        self.unknown_face_data.append({
                            'id': unknown_id,
                            'assigned_name': None,
                            'first_seen': datetime.now().isoformat(),
                            'last_seen': datetime.now().isoformat(),
                            'visit_count': 1
                        })

                        confidence = 100  # New unknown face
                        face_type = "new_unknown"

                        # Track detection
                        self.current_detections[person_id] = True
                        self.detection_start_time[person_id] = current_time
                        self.log_activity(name, "new_unknown", "first_detection", confidence)

                        # Play sound
                        self.play_sound("unknown")
                        self.last_sound_time[person_id] = current_time

                        logger.info(f"New unknown person detected: {name}")

                recognized_faces.append({
                    'name': name,
                    'location': face_location,
                    'confidence': confidence,
                    'face_type': face_type,
                    'unknown_id': unknown_id,
                    'encoding': face_encoding,
                    'person_id': person_id
                })

            return recognized_faces

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return []

    def draw_face_boxes(self, frame, recognized_faces):
        """Draw bounding boxes and labels on detected faces"""
        for face_data in recognized_faces:
            top, right, bottom, left = face_data['location']
            name = face_data['name']
            confidence = face_data['confidence']
            face_type = face_data['face_type']

            # Choose color based on face type
            if face_type == "known":
                color = (0, 255, 0)  # Green for known
            elif face_type == "repeat_unknown":
                color = (0, 255, 255)  # Yellow for repeat unknown
            else:
                color = (0, 0, 255)  # Red for new unknown

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Prepare label
            if face_type == "known":
                label = f"{name} ({confidence:.1f}%)"
            elif face_type == "repeat_unknown":
                # Find visit count for this unknown face
                visits = 1
                for data in self.unknown_face_data:
                    if data['id'] == face_data['unknown_id']:
                        visits = data['visit_count']
                        break
                label = f"{name} (visits: {visits})"
            else:
                label = f"{name} (NEW)"

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
            )

            # Draw label background
            cv2.rectangle(frame, (left, bottom - text_height - 10),
                          (left + text_width, bottom), color, cv2.FILLED)

            # Draw label text
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def save_capture(self, frame, recognized_faces):
        """Save frame with timestamp when faces are detected (with rate limiting)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, face_data in enumerate(recognized_faces):
            name = face_data['name']
            face_type = face_data['face_type']
            person_id = face_data['person_id']
            confidence = face_data['confidence']

            # Check if we should capture for this person
            if not self.should_capture(person_id):
                continue

            # Update last capture time
            self.last_capture_time[person_id] = time.time()

            # Create filename based on face type
            if face_type == "known":
                filename = f"{timestamp}_{name}_known_{i}.jpg"
            else:
                filename = f"{timestamp}_{name}_{face_type}_{i}.jpg"

            filepath = os.path.join(self.output_path, filename)

            success = cv2.imwrite(filepath, frame)
            if success:
                logger.info(f"Saved capture: {filename}")

                # Log capture to database
                self.log_capture_to_db(name, face_type, filename, confidence)

                # Also save unknown face crop for easier identification
                if face_type in ["new_unknown", "repeat_unknown"]:
                    self.save_face_crop(frame, face_data, timestamp)
            else:
                logger.error(f"Failed to save capture: {filename}")

    def log_capture_to_db(self, person_name, face_type, filename, confidence):
        """Log capture to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO captures (person_name, face_type, filename, confidence)
                VALUES (?, ?, ?, ?)
            ''', (person_name, face_type, filename, confidence))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log capture to database: {e}")

    def save_face_crop(self, frame, face_data, timestamp):
        """Save cropped face image for unknown persons"""
        try:
            top, right, bottom, left = face_data['location']

            # Add padding around face
            padding = 20
            height, width = frame.shape[:2]

            top = max(0, top - padding)
            bottom = min(height, bottom + padding)
            left = max(0, left - padding)
            right = min(width, right + padding)

            # Crop face
            face_crop = frame[top:bottom, left:right]

            # Save cropped face
            crop_filename = f"{timestamp}_{face_data['name']}_crop.jpg"
            crop_filepath = os.path.join(self.unknown_faces_path, crop_filename)

            cv2.imwrite(crop_filepath, face_crop)
            logger.info(f"Saved face crop: {crop_filename}")

        except Exception as e:
            logger.error(f"Error saving face crop: {str(e)}")

    def assign_name_to_unknown(self, unknown_id, name):
        """Assign a name to an unknown person"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Update unknown face with assigned name
            cursor.execute('''
                UPDATE unknown_faces 
                SET assigned_name = ? 
                WHERE id = ?
            ''', (name, unknown_id))

            # Get the encoding to add to known faces
            cursor.execute('SELECT encoding FROM unknown_faces WHERE id = ?', (unknown_id,))
            result = cursor.fetchone()

            if result:
                encoding_blob = result[0]
                encoding = pickle.loads(encoding_blob)

                # Add to known faces
                cursor.execute('''
                    INSERT OR REPLACE INTO known_faces (name, encoding)
                    VALUES (?, ?)
                ''', (name, encoding_blob))

                # Add to local arrays
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

                # Log the assignment
                self.log_activity(name, "unknown_assigned", "name_assigned",
                                  additional_info=f"Unknown ID: {unknown_id}")

                conn.commit()
                logger.info(f"Successfully assigned name '{name}' to Unknown_{unknown_id}")

            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error assigning name to unknown person: {str(e)}")
            return False

    def list_unknown_faces(self):
        """List all unknown faces with their information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, assigned_name, first_seen, last_seen, visit_count
                FROM unknown_faces 
                WHERE is_active = 1
                ORDER BY first_seen
            ''')

            rows = cursor.fetchall()
            conn.close()

            print("\n=== Unknown Faces Database ===")
            if not rows:
                print("No unknown faces in database")
                return

            for row in rows:
                unknown_id, assigned_name, first_seen, last_seen, visit_count = row
                assigned_name = assigned_name or 'Not assigned'
                print(f"ID: {unknown_id}")
                print(f"  Assigned Name: {assigned_name}")
                print(f"  First Seen: {first_seen}")
                print(f"  Last Seen: {last_seen}")
                print(f"  Visit Count: {visit_count}")
                print("-" * 30)

        except Exception as e:
            logger.error(f"Error listing unknown faces: {str(e)}")

    def display_activity_summary(self):
        """Display today's activity summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT person_name, face_type, action, timestamp, confidence
                FROM activity_log 
                WHERE date(timestamp) = date('now')
                ORDER BY timestamp DESC
                LIMIT 20
            ''')

            rows = cursor.fetchall()
            conn.close()

            print(f"\n=== Today's Activity Summary ({len(rows)} recent events) ===")

            for row in rows:
                person_name, face_type, action, timestamp, confidence = row
                confidence_str = f"({confidence:.1f}%)" if confidence else ""
                print(f"{timestamp} - {person_name} {confidence_str} - {action} ({face_type})")

        except Exception as e:
            logger.error(f"Error displaying activity summary: {str(e)}")

    def get_valid_camera_index(self, max_index=5):
        """Find and return the first available camera index"""
        for index in range(max_index):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.release()
                return index
            cap.release()
        return None

    def run_camera_detection(self, camera_index=1, save_captures=True):
        """Run real-time face detection from camera"""
        try:
            if camera_index is None:
                camera_index = self.get_valid_camera_index()
                if camera_index is None:
                    logger.error("No available camera found")
                    return

            cap = cv2.VideoCapture(1)

            if not cap.isOpened():
                logger.error(f"Cannot open camera {camera_index}")
                return

            frame_count = 0
            process_every_n_frames = 1  # Process every frame for constant detection

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            print("\n=== Parking Lot Face Recognition System ===")
            print("Controls:")
            print("- Press 'q' to quit")
            print("- Press 's' to manually save current frame")
            print("- Press 'l' to list unknown faces")
            print("- Press 'a' to assign name to unknown person")
            print("- Press 'r' to reload known faces")
            print("- Press 'v' to view today's activity summary")
            print(f"\nSound notifications: {'Enabled' if SOUND_AVAILABLE else 'Disabled'}")
            print(f"Capture interval: {'Continuous (no delay)' if self.capture_interval == 0 else f'{self.capture_interval} seconds per person'}")
            print(f"Sound interval: {'Continuous (no delay)' if self.sound_interval == 0 else f'{self.sound_interval} seconds per person'}")
            print("CONSTANT DETECTION MODE: Processing every frame with no delays")
            logger.info(f"Starting camera detection on camera {camera_index}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                frame_count += 1

                if frame_count % process_every_n_frames == 0:
                    recognized_faces = self.recognize_face(frame)
                    frame = self.draw_face_boxes(frame, recognized_faces)

                    if save_captures and recognized_faces:
                        self.save_capture(frame, recognized_faces)

                    self.update_activity_tracking(recognized_faces)

                cv2.imshow('Parking Lot Face Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_capture_{timestamp}.jpg"
                    filepath = os.path.join(self.output_path, filename)
                    cv2.imwrite(filepath, frame)
                    logger.info(f"Manual capture saved: {filename}")
                elif key == ord('v'):
                    self.display_activity_summary()
                elif key == ord('l'):
                    self.list_unknown_faces()
                elif key == ord('a'):
                    self.interactive_name_assignment()

            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera detection stopped")

        except Exception as e:
            logger.error(f"Error in camera detection: {str(e)}")
        finally:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass

    def update_activity_tracking(self, recognized_faces):
        """Update activity tracking for continuous detections"""
        current_time = time.time()
        current_person_ids = set()

        # Update currently detected faces
        for face_data in recognized_faces:
            person_id = face_data['person_id']
            current_person_ids.add(person_id)

        # Check for people who are no longer detected
        for person_id in list(self.current_detections.keys()):
            if person_id not in current_person_ids:
                # Person is no longer detected
                detection_duration = current_time - self.detection_start_time[person_id]

                if detection_duration >= self.min_detection_duration:
                    # Log visit end
                    name = person_id.split('_', 1)[1]  # Extract name from person_id
                    face_type = "known" if person_id.startswith("known_") else "unknown"

                    self.log_activity(name, face_type, "visit_ended",
                                      additional_info=f"Duration: {detection_duration:.1f}s")

                    logger.info(f"Visit ended: {name} (duration: {detection_duration:.1f}s)")

                # Remove from tracking
                del self.current_detections[person_id]
                del self.detection_start_time[person_id]

    def interactive_name_assignment(self):
        """Interactive function to assign names to unknown persons"""
        try:
            # List unknown faces first
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, assigned_name, first_seen, last_seen, visit_count
                FROM unknown_faces 
                WHERE is_active = 1 AND assigned_name IS NULL
                ORDER BY visit_count DESC, first_seen
            ''')

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                print("No unassigned unknown faces found")
                return

            print("\n=== Unassigned Unknown Faces ===")
            for row in rows:
                unknown_id, _, first_seen, last_seen, visit_count = row
                print(f"ID: {unknown_id} - Visits: {visit_count} - First: {first_seen}")

            # Get user input
            try:
                unknown_id = int(input("Enter Unknown ID to assign name (or 0 to cancel): "))
                if unknown_id == 0:
                    return

                # Check if ID exists
                if not any(row[0] == unknown_id for row in rows):
                    print("Invalid Unknown ID")
                    return

                name = input("Enter name for this person: ").strip()
                if not name:
                    print("Name cannot be empty")
                    return

                # Assign name
                if self.assign_name_to_unknown(unknown_id, name):
                    print(f"Successfully assigned name '{name}' to Unknown_{unknown_id}")
                else:
                    print("Failed to assign name")

            except ValueError:
                print("Invalid input")

        except Exception as e:
            logger.error(f"Error in interactive name assignment: {str(e)}")

    def process_video_file(self, video_path, save_captures=True):
        """Process a video file for face recognition"""
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
            process_every_n_frames = int(fps)  # Process one frame per second

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process every N frames
                if frame_count % process_every_n_frames == 0:
                    recognized_faces = self.recognize_face(frame)

                    if recognized_faces:
                        frame = self.draw_face_boxes(frame, recognized_faces)

                        if save_captures:
                            self.save_capture(frame, recognized_faces)

                        # Display progress
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"Progress: {progress:.1f}% - Found {len(recognized_faces)} faces")

                # Display frame (optional)
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video processing completed")

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")

    def generate_report(self, days=7):
        """Generate a comprehensive report of parking lot activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            print(f"\n=== Parking Lot Activity Report (Last {days} days) ===")
            print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            # Known persons activity
            cursor.execute('''
                SELECT person_name, COUNT(*) as visits, 
                       MIN(timestamp) as first_visit, MAX(timestamp) as last_visit
                FROM activity_log 
                WHERE face_type = 'known' 
                AND timestamp >= datetime('now', '-{} days')
                GROUP BY person_name
                ORDER BY visits DESC
            '''.format(days))

            known_data = cursor.fetchall()

            print(f"\nKnown Persons Activity:")
            print("-" * 40)
            if known_data:
                for person_name, visits, first_visit, last_visit in known_data:
                    print(f"{person_name}: {visits} visits")
                    print(f"  First: {first_visit}")
                    print(f"  Last: {last_visit}")
            else:
                print("No known person activity found")

            # Unknown persons summary
            cursor.execute('''
                SELECT COUNT(*) as total_unknown, 
                       COUNT(CASE WHEN assigned_name IS NOT NULL THEN 1 END) as assigned,
                       SUM(visit_count) as total_visits
                FROM unknown_faces 
                WHERE is_active = 1
            ''')

            unknown_summary = cursor.fetchone()
            total_unknown, assigned, total_visits = unknown_summary

            print(f"\nUnknown Persons Summary:")
            print("-" * 40)
            print(f"Total Unknown Faces: {total_unknown}")
            print(f"Assigned Names: {assigned}")
            print(f"Unassigned: {total_unknown - assigned}")
            print(f"Total Unknown Visits: {total_visits}")

            # Top unknown visitors
            cursor.execute('''
                SELECT id, assigned_name, visit_count, first_seen, last_seen
                FROM unknown_faces 
                WHERE is_active = 1
                ORDER BY visit_count DESC
                LIMIT 10
            ''')

            top_unknown = cursor.fetchall()

            print(f"\nTop Unknown Visitors:")
            print("-" * 40)
            for unknown_id, assigned_name, visits, first_seen, last_seen in top_unknown:
                name = assigned_name or f"Unknown_{unknown_id}"
                print(f"{name}: {visits} visits")
                print(f"  First: {first_seen}")
                print(f"  Last: {last_seen}")

            # Daily activity summary
            cursor.execute('''
                SELECT date(timestamp) as day, COUNT(*) as activities
                FROM activity_log 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY date(timestamp)
                ORDER BY day DESC
            '''.format(days))

            daily_activity = cursor.fetchall()

            print(f"\nDaily Activity Summary:")
            print("-" * 40)
            for day, activities in daily_activity:
                print(f"{day}: {activities} activities")

            conn.close()

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")

    def cleanup_old_data(self, days_to_keep=30):
        """Clean up old activity logs and captures"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clean up old activity logs
            cursor.execute('''
                DELETE FROM activity_log 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))

            deleted_logs = cursor.rowcount

            # Clean up old captures
            cursor.execute('''
                DELETE FROM captures 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))

            deleted_captures = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Cleanup completed: {deleted_logs} logs, {deleted_captures} captures removed")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Global instance for use by main.py
_face_recognition_system = None

def get_face_recognition_system():
    """Get or create the global face recognition system"""
    global _face_recognition_system
    if _face_recognition_system is None:
        _face_recognition_system = ParkingLotFaceRecognition()
    return _face_recognition_system

def recognize_face_id(frame):
    """Main function to be called by main.py - returns recognized face name"""
    try:
        system = get_face_recognition_system()
        recognized_faces = system.recognize_face(frame)
        
        # Return the first known face found, or None if no faces detected
        for face_data in recognized_faces:
            if isinstance(face_data, dict) and 'name' in face_data and face_data['name'] != "Unknown":
                return face_data['name']
        
        return None
    except Exception as e:
        logger.error(f"Error in recognize_face_id: {e}")
        return None

def main():
    """Main function with command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Parking Lot Face Recognition System')
    parser.add_argument('--mode', choices=['camera', 'video', 'report', 'list', 'cleanup'],
                        default='camera', help='Operation mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--no-save', action='store_true', help='Disable saving captures')
    parser.add_argument('--days', type=int, default=7, help='Number of days for report/cleanup')

    args = parser.parse_args()

    # Initialize the system
    recognition_system = ParkingLotFaceRecognition()

    try:
        if args.mode == 'camera':
            recognition_system.run_camera_detection(
                camera_index=args.camera,
                save_captures=not args.no_save
            )
        elif args.mode == 'video':
            if not args.video:
                print("Error: --video path is required for video mode")
                return
            recognition_system.process_video_file(
                args.video,
                save_captures=not args.no_save
            )
        elif args.mode == 'report':
            recognition_system.generate_report(days=args.days)
        elif args.mode == 'list':
            recognition_system.list_unknown_faces()
        elif args.mode == 'cleanup':
            recognition_system.cleanup_old_data(days_to_keep=args.days)

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.info("Program ended")

if __name__ == "__main__":
    main()