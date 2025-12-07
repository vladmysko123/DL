from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

def main():
    # 1. Download LFW (first time it will actually download, later it uses cache)
    # min_faces_per_person=20 -> keep only people with >=20 images
    print("Downloading LFW dataset (first run may take some time)...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)

    images = lfw.images
    targets = lfw.target
    target_names = lfw.target_names

    print("Dataset loaded.")
    print("Images shape:", images.shape)      # (n_samples, h, w)
    print("Number of people:", len(target_names))

    # 2. Show a few sample faces with names
    n_examples = 6
    plt.figure(figsize=(10, 4))
    for i in range(n_examples):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        person_name = target_names[targets[i]]
        plt.title(person_name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 3. (Optional) Show how many images per person
    import numpy as np
    counts = [(target_names[i], (targets == i).sum()) for i in range(len(target_names))]
    counts_sorted = sorted(counts, key=lambda x: x[1], reverse=True)

    print("\nTop 10 people by number of images:")
    for name, cnt in counts_sorted[:10]:
        print(f"{name}: {cnt} images")

if __name__ == "__main__":
    main()
