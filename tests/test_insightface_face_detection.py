# test_detect.py
from pathlib import Path
import cv2
import numpy as np

from blurx.utils.io import iter_images
from blurx.models.face_detection import insightface_face_detection

# <<< EDIT THIS >>>
INPUT_PATH = (
    "/Users/montasardridi/Documents/scorpia/blurX/data/multiple_faces_clear_image.jpg"
)
OUT_DIR = Path("detected_outputs")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for img_path, img in iter_images(INPUT_PATH):
    faces = insightface_face_detection(img)
    print(f"{img_path}: {len(faces)} face(s)")

    annotated = img.copy()
    h, w = annotated.shape[:2]

    for face in faces:
        # bbox: [x1, y1, x2, y2, score]
        x1, y1, x2, y2 = [int(round(v)) for v in face.bbox[:4]]

        # clip to bounds
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # optional: draw 5 keypoints if present
        if hasattr(face, "kps"):
            kps = np.asarray(face.kps, dtype=float)
            for kx, ky in kps:
                kxi = max(0, min(w - 1, int(round(kx))))
                kyi = max(0, min(h - 1, int(round(ky))))
                cv2.circle(annotated, (kxi, kyi), 2, (255, 0, 0), -1)

    out_path = OUT_DIR / (Path(img_path).stem + "_detected.jpg")
    ok = cv2.imwrite(str(out_path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to write {out_path}")
    print(f"Saved -> {out_path}")
