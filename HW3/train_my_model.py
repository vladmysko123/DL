import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

DATASET = "dataset_cropped"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MTCNN for face alignment (optional but improves quality)
    mtcnn = MTCNN(image_size=160, margin=0, device=device)

    # Pretrained FaceNet model → outputs embeddings
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    embeddings = []
    labels = []

    print("Extracting embeddings...")

    for person in os.listdir(DATASET):
        person_folder = os.path.join(DATASET, person)
        if not os.path.isdir(person_folder):
            continue

        for file in os.listdir(person_folder):
            if not file.lower().endswith(('.jpg', '.png')):
                continue

            path = os.path.join(person_folder, file)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            # Use MTCNN to detect & align face
            face = mtcnn(img)
            if face is None:
                print("No face:", path)
                continue

            face = face.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(face).cpu().numpy().flatten()

            embeddings.append(embedding)
            labels.append(person)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print("Training classifier...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, y)

    print("Saving model...")
    joblib.dump(clf, "pytorch_classifier.pkl")
    joblib.dump(encoder, "pytorch_labelencoder.pkl")

    np.save("embeddings.npy", embeddings)

    print("Training complete! Trained on", len(embeddings), "images.")

if __name__ == "__main__":
    main()
