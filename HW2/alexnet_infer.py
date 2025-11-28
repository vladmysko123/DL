from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd


# ---------- Custom dataset for test ----------
class ButterflyTestDataset(Dataset):
    def __init__(self, df, root_dir: Path, path_col: str, id_col: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.path_col = path_col
        self.id_col = id_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        img_path = self.root_dir / rel_path
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        img_id = row[self.id_col]  # just the id (int or str)
        return img, img_id


def main():
    # ---------- Config ----------
    DATA_DIR = Path("lnu-butterflies")  # CHANGE if needed
    TEST_DIR = DATA_DIR / "test"
    CLASSES_TXT = DATA_DIR / "classes.txt"
    BEST_MODEL_PATH = Path("models/alexnet_butterflies_best.pth")
    SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
    TEST_CSV = DATA_DIR / "test.csv"

    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    # ---------- Classes (for mapping index -> class name) ----------
    classes = [line.strip() for line in CLASSES_TXT.read_text().splitlines() if line.strip()]
    num_classes = len(classes)
    print("Loaded", num_classes, "classes from classes.txt")

    # ---------- Test dataframe ----------
    test_df = pd.read_csv(TEST_CSV)
    print("test_df columns:", test_df.columns.tolist())

    # Guess column names – adjust here if different:
    ID_COL = "id"      # CHANGE if your id column has another name
    PATH_COL = "path"  # CHANGE if test.csv actually stores relative paths

    if PATH_COL not in test_df.columns:
        # if test.csv only has id and files are `test/<id>.jpg`
        test_df[PATH_COL] = test_df[ID_COL].astype(str).apply(lambda x: f"test/{x}.jpg")

    # ---------- Transforms ----------
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------- Dataset & DataLoader ----------
    test_dataset = ButterflyTestDataset(
        test_df, DATA_DIR, PATH_COL, ID_COL, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # simpler & safe on Windows
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # ---------- Build AlexNet & load weights ----------
    try:
        from torchvision.models import AlexNet_Weights
        model = models.alexnet(weights=None)
    except Exception:
        model = models.alexnet(pretrained=False)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # ---------- Inference ----------
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(DEVICE, non_blocking=True)

            outputs = model(images)
            _, pred_idx = torch.max(outputs, 1)
            pred_idx = pred_idx.cpu().numpy()

            # ids is either a list of strings or a tensor of ints
            for i, img_id in enumerate(ids):
                if not isinstance(img_id, str):
                    # likely a tensor scalar
                    img_id = img_id.item()
                cls = classes[pred_idx[i]]
                all_ids.append(img_id)
                all_preds.append(cls)

    # ---------- Build submission ----------
    sample_sub = pd.read_csv(SAMPLE_SUB)
    print("sample_submission columns:", sample_sub.columns.tolist())

    # assume first col is id, second is label column
    sub_id_col = sample_sub.columns[0]
    sub_label_col = sample_sub.columns[1]

    submission = pd.DataFrame({
        sub_id_col: all_ids,
        sub_label_col: all_preds,
    })

    submission_path = Path("submission_alexnet.csv")
    submission.to_csv(submission_path, index=False)
    print("Saved submission to:", submission_path)


if __name__ == "__main__":
    main()
