import os
import cv2
import shutil
from sklearn.datasets import fetch_lfw_people

DATASET_DIR = "dataset"

def main():
    print("Downloading raw LFW dataset...")
    
    lfw = fetch_lfw_people(
        min_faces_per_person=20,
        resize=1.0,
        color=True,
        funneled=False,   # THIS IS IMPORTANT — preserves real image data
        slice_=None
    )

    images = lfw.images       # uint8 images
    targets = lfw.target
    names = lfw.target_names

    # Remove old dataset folder if exists
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

    print("Saving dataset...")

    for idx, img in enumerate(images):
        name = names[targets[idx]]
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        folder = os.path.join(DATASET_DIR, name)
        os.makedirs(folder, exist_ok=True)

        filename = f"{len(os.listdir(folder)):03d}.jpg"
        path = os.path.join(folder, filename)

        # img is already uint8 RGB, need to convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)

    print("Done! Dataset saved to:", DATASET_DIR)

if __name__ == "__main__":
    main()
