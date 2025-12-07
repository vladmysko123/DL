import os
import cv2
import shutil

INPUT_DIR = "dataset"            # your manually created dataset
OUTPUT_DIR = "dataset_cropped"   # will be generated automatically

# Haar cascade (included with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Remove old cropped folder if exists
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("Cropping faces...")

    for person in os.listdir(INPUT_DIR):
        person_folder = os.path.join(INPUT_DIR, person)
        if not os.path.isdir(person_folder):
            continue

        output_person_folder = os.path.join(OUTPUT_DIR, person)
        os.makedirs(output_person_folder, exist_ok=True)

        for img_name in os.listdir(person_folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print("Cannot read:", img_path)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) == 0:
                print("No face detected:", img_path)
                continue

            # take the biggest detected face (in case multiple)
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

            # crop & resize
            face_crop = img[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (160, 160))  # suitable for DeepFace models

            out_name = f"{len(os.listdir(output_person_folder)):03d}.jpg"
            out_path = os.path.join(output_person_folder, out_name)
            cv2.imwrite(out_path, face_crop)

            print("Saved:", out_path)

    print("\nDone! Cropped faces saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
