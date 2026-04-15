import cv2
import numpy as np
import math
import csv
import os
from pathlib import Path

# point this to wherever you extracted the dataset
DATASET_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\AD BACKEND\animal-atc-backend\Indian_bovine_breeds\Indian_bovine_breeds"
OUTPUT_CSV = "breed_data.csv"

def extract_features(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    if cw == 0 or ch == 0:
        return None

    # measurements
    px_to_cm = 150.0 / cw
    body_length = round(cw * px_to_cm, 1)
    height_withers = round(ch * 0.85 * px_to_cm, 1)
    chest_width = round(cw * 0.45 * px_to_cm, 1)

    body_length    = max(120, min(180, body_length))
    height_withers = max(120, min(155, height_withers))
    chest_width    = max(55,  min(85,  chest_width))

    aspect_ratio = round(cw / ch, 2)
    if aspect_ratio > 1.8:
        height_withers = min(155, height_withers * 1.15)
    elif aspect_ratio < 1.2:
        height_withers = max(120, height_withers * 0.9)
    height_withers = round(height_withers, 1)

    # rump angle
    rump_angle = 10.0
    top_points = [(pt[0][0], pt[0][1]) for pt in largest if pt[0][1] < y + ch * 0.4]
    if len(top_points) >= 2:
        top_points.sort(key=lambda p: p[0])
        rump_pts = top_points[int(len(top_points) * 0.7):]
        if len(rump_pts) >= 2:
            dx = rump_pts[-1][0] - rump_pts[0][0]
            dy = rump_pts[-1][1] - rump_pts[0][1]
            if dx != 0:
                rump_angle = round(abs(math.degrees(math.atan2(dy, dx))), 1)
                rump_angle = max(3, min(25, rump_angle))

    # bcs
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    animal_region = hsv[y:y+ch, x:x+cw]
    mean_saturation = float(np.mean(animal_region[:, :, 1]))
    bcs = round(1.0 + (mean_saturation / 255.0) * 4.0, 1)
    bcs = max(1.0, min(5.0, bcs))

    return {
        "body_length":    body_length,
        "height_withers": height_withers,
        "chest_width":    chest_width,
        "rump_angle":     rump_angle,
        "bcs":            bcs,
        "aspect_ratio":   aspect_ratio,
    }


def build_dataset():
    dataset_path = Path(DATASET_PATH)
    breeds = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(breeds)} breeds")

    rows = []
    for breed_folder in breeds:
        breed_name = breed_folder.name.replace("_", " ")
        images = list(breed_folder.glob("*.jpg")) + \
                 list(breed_folder.glob("*.jpeg")) + \
                 list(breed_folder.glob("*.png"))

        print(f"Processing {breed_name}: {len(images)} images")
        success = 0
        for img_path in images:
            features = extract_features(img_path)
            if features:
                features["breed"] = breed_name
                rows.append(features)
                success += 1

        print(f"  → {success} successful")

    # write csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "body_length", "height_withers", "chest_width",
            "rump_angle", "bcs", "aspect_ratio", "breed"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} total rows saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    build_dataset()