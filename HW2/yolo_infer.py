from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ---------- Paths ----------
DATA_DIR = Path("lnu-butterflies")
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# path to your trained YOLO weights
WEIGHTS = Path(
    r"D:\botmakers\projects\instafill\processing-api-instafill\runs\classify\train6\weights\best.pt"
)

# ---------- Load model ----------
model = YOLO(str(WEIGHTS))

# ---------- Load test info ----------
test_df = pd.read_csv(TEST_CSV)
print("test_df columns:", test_df.columns.tolist())  # should be ['id', 'path']

# ---------- Run predictions ----------
ids = []
preds = []

for _, row in test_df.iterrows():
    img_id = row["id"]
    img_rel_path = row["path"]          # e.g. "test/0001.jpg"
    img_path = DATA_DIR / img_rel_path  # lnu-butterflies/test/0001.jpg

    # YOLO predict returns a list; take first Result
    result = model.predict(
        source=str(img_path),
        imgsz=224,
        verbose=False
    )[0]

    top1 = int(result.probs.top1)         # predicted class index
    class_name = result.names[top1]       # map to class name string

    ids.append(img_id)
    preds.append(class_name)

# ---------- Build submission in Kaggle format ----------
sample = pd.read_csv(SAMPLE_SUB)
sub_id_col = sample.columns[0]     # usually "id"
sub_label_col = sample.columns[1]  # usually "target" or "label"

submission = pd.DataFrame({
    sub_id_col: ids,
    sub_label_col: preds,
})

out_path = Path("submission_yolo.csv")
submission.to_csv(out_path, index=False)
print("Saved submission to:", out_path.resolve())
