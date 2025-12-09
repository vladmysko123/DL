import cv2
import face_recognition
import numpy as np
from encode_faces_helper import load_known_faces

THRESHOLD = 0.6 

def distance_to_confidence(distance, threshold=THRESHOLD):
    """
    Converts face distance to a probability-like confidence score [0-1].
    0 → perfect match
    threshold → 0 confidence
    """
    if distance > threshold:
        return 0.0
    return float(1.0 - distance / threshold)

def main():
    print("Loading known faces...")
    known_face_encodings, known_face_names = load_known_faces()
    print(f"Loaded {len(known_face_encodings)} encodings for {len(set(known_face_names))} users.\n")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame from camera")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            name = "Unknown"
            color = (0, 0, 255)  
            probability_text = "0%"

            if len(known_face_encodings) > 0:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                confidence = distance_to_confidence(best_dist)

                probability_text = f"{confidence * 100:.1f}%"

                if best_dist < THRESHOLD:
                    color = (0, 255, 0) 
                    name = known_face_names[best_idx]

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            label = f"{name} ({probability_text})"
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face Recognition — Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
