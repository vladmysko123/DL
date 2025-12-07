import cv2
import torch
import numpy as np
import joblib
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained models
print("Loading FaceNet model...")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Loading classifier...")
clf = joblib.load("pytorch_classifier.pkl")
encoder = joblib.load("pytorch_labelencoder.pkl")

def get_embedding(frame_rgb):
    """
    Takes an RGB frame, extracts aligned face using MTCNN,
    and generates a 512-dim embedding using FaceNet.
    """
    face = mtcnn(frame_rgb)
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face).cpu().numpy().flatten()

    return embedding


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        embedding = get_embedding(rgb)

        if embedding is not None:
            probs = clf.predict_proba([embedding])[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            name = encoder.classes_[idx]

            # Decide rectangle color
            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)

            # Draw info
            cv2.putText(frame, f"{name} ({conf*100:.1f}%)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 2)

        else:
            cv2.putText(frame, "No face detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        cv2.imshow("PyTorch Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
