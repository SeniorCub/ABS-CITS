import os
import datetime

def log_access(face_id, plate_text, access):
    with open("access_log.txt", "a") as f:
        status = "GRANTED" if access else "DENIED"
        f.write(f"{datetime.datetime.now()} - {face_id} - {plate_text} - {status}\n")

def load_known_faces(folder):
    known = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            known[name] = os.path.join(folder, filename)
    return known

def load_known_plates(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
