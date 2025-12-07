import os
import face_recognition

DATASET_DIR = "dataset"

def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_folder, img_name)
            image = face_recognition.load_image_file(img_path)

            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                continue

            encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

            known_face_encodings.append(encoding)
            known_face_names.append(person_name)

    return known_face_encodings, known_face_names
