import cv2
import numpy as np
import math
import base64
import pickle
import os
import tensorflow as tf
import json

def decode_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred

def find_animal(img):
    h, w = img.shape[:2]

    # Step 1: try GrabCut to segment foreground (the animal) from background
    mask = np.zeros((h, w), np.uint8)
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        grab_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    except Exception:
        grab_mask = np.ones((h, w), dtype="uint8") * 255

    # Step 2: morphological cleanup — close small holes, remove small blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    grab_mask = cv2.morphologyEx(grab_mask, cv2.MORPH_CLOSE, kernel)
    grab_mask = cv2.morphologyEx(grab_mask, cv2.MORPH_OPEN, kernel)

    # Step 3: find contours on the cleaned mask
    contours, _ = cv2.findContours(grab_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: filter — must be at least 8% of image area
    min_area = h * w * 0.08
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Step 5: fallback to Canny if GrabCut found nothing big enough
    if not large_contours:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blurred, 30, 120)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        edge = cv2.dilate(edge, kernel2, iterations=2)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel2)
        contours2, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours2 if cv2.contourArea(c) > min_area]
        if not large_contours:
            large_contours = contours2 if contours2 else contours

    if not large_contours:
        raise ValueError("No animal detected in the image")

    largest = max(large_contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Step 6: sanity check — bounding box must be at least 15% of image in both dims
    if cw < w * 0.15 or ch < h * 0.15:
        # box is too small, just use center 80% of image as fallback
        x = int(w * 0.1)
        y = int(h * 0.1)
        cw = int(w * 0.8)
        ch = int(h * 0.8)
        largest = np.array([[[x, y]], [[x+cw, y]], [[x+cw, y+ch]], [[x, y+ch]]])

    return largest, x, y, cw, ch

def extract_measurements(largest, x, y, cw, ch):
    px_to_cm = 150.0 / cw
    body_len = round(cw * px_to_cm, 1)
    height_withers = round(ch * 0.85 * px_to_cm, 1)
    chest_width = round(cw * 0.45 * px_to_cm, 1)
    body_len = max(120, min(180, body_len))
    height_withers = max(120, min(155, height_withers))
    chest_width = max(55, min(85, chest_width))
    aspect_ratio = cw / ch if ch > 0 else 1.0
    if aspect_ratio > 1.8:
        height_withers = min(155, height_withers * 1.15)
    elif aspect_ratio < 1.2:
        height_withers = max(120, height_withers * 0.9)
    height_withers = round(height_withers, 1)
    top_points = [(pt[0][0], pt[0][1]) for pt in largest if pt[0][1] < y + ch * 0.4]
    rump_angle = 10.0
    if len(top_points) >= 2:
        top_points.sort(key=lambda p: p[0])
        rump_pts = top_points[int(len(top_points) * 0.7):]
        if len(rump_pts) >= 2:
            dx = rump_pts[-1][0] - rump_pts[0][0]
            dy = rump_pts[-1][1] - rump_pts[0][1]
            if dx != 0:
                rump_angle = round(abs(math.degrees(math.atan2(dy, dx))), 1)
                rump_angle = max(3, min(25, rump_angle))
    return body_len, height_withers, chest_width, rump_angle

def calculate_score(body_length, height_withers, chest_width, rump_angle):
    score = 0
    if 140 <= body_length <= 160:
        score += 30
    elif 130 <= body_length <= 170:
        score += 20
    else:
        score += 10
    if 130 <= height_withers <= 145:
        score += 25
    elif 120 <= height_withers <= 155:
        score += 15
    else:
        score += 8
    if 65 <= chest_width <= 80:
        score += 25
    elif 55 <= chest_width <= 85:
        score += 15
    else:
        score += 8
    if 5 <= rump_angle <= 15:
        score += 20
    elif 3 <= rump_angle <= 20:
        score += 12
    else:
        score += 5
    return min(score, 100)

_cnn_model = None
_cnn_class_names = None

def _get_cnn_model():
    global _cnn_model, _cnn_class_names
    if _cnn_model is None:
        model_path = os.path.join(os.path.dirname(__file__), "best_cnn_model.keras")
        names_path = os.path.join(os.path.dirname(__file__), "cnn_class_names.json")
        if os.path.exists(model_path) and os.path.exists(names_path):
            _cnn_model = tf.keras.models.load_model(model_path)
            with open(names_path, "r") as f:
                _cnn_class_names = json.load(f)
            print("CNN model loaded successfully")
        else:
            print("CNN model not found, using fallback rules")
    return _cnn_model, _cnn_class_names

def estimate_breed(img, x, y, cw, ch, body_len, height_withers, chest_width, rump_angle, bcs):
    model, class_names = _get_cnn_model()
    if model is not None:
        full_resized = cv2.resize(img, (224, 224))
        full_resized = full_resized.astype("float32") / 255.0
        full_resized = np.expand_dims(full_resized, axis=0)
        predictions = model.predict(full_resized, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        breed = class_names[str(predicted_index)]
        if "indian_bovine_breeds" in breed.lower():
            predictions[0][predicted_index] = 0
            predicted_index = int(np.argmax(predictions[0]))
            breed = class_names[str(predicted_index)]
        confidence = predictions[0][predicted_index] * 100
        # get top 3 predictions
        top3 = predictions[0].argsort()[-3:][::-1]
        top3_result = " | ".join([
            f"{class_names[str(i)]} ({predictions[0][i]*100:.0f}%)"
            for i in top3
            if "indian_bovine_breeds" not in class_names[str(i)].lower()
        ])
        print(f"Top 3: {top3_result}")
        return f"{breed} ({confidence:.0f}%)"
    else:
        if rump_angle <= 5:
            return "Murrah"
        elif rump_angle <= 13:
            return "Sahiwal"
        elif height_withers >= 136 and bcs < 2.0:
            return "Holstein Friesian"
        elif height_withers >= 136 and bcs >= 2.0:
            return "Gir"
        elif height_withers < 130 and bcs < 1.6:
            return "Surti"
        else:
            return "HF-Cross"
 

def calculate_bcs(img, x, y, cw, ch):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    animal_region = hsv[y:y+ch, x:x+cw]
    mean_saturation = float(np.mean(animal_region[:, :, 1]))
    bcs = round(1 + (mean_saturation / 255) * 4, 1)
    bcs = max(1, min(5, bcs))
    return bcs

def annotate_image(img, largest, x, y, cw, ch):
    annotated = img.copy()
    cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 100), 2)
    keypoints = {
        "Withers": (x + int(cw * 0.25), y + int(ch * 0.15)),
        "Back":    (x + int(cw * 0.45), y + int(ch * 0.12)),
        "Hip":     (x + int(cw * 0.70), y + int(ch * 0.15)),
        "Rump":    (x + int(cw * 0.85), y + int(ch * 0.35)),
        "Chest":   (x + int(cw * 0.20), y + int(ch * 0.55)),
        "Barrel":  (x + int(cw * 0.65), y + int(ch * 0.55)),
    }
    for name, (kx, ky) in keypoints.items():
        cv2.circle(annotated, (kx, ky), 6, (0, 200, 255), -1)
        cv2.putText(annotated, name, (kx + 8, ky - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    connections = [(0,1),(1,2),(2,3),(0,4),(2,5),(4,5)]
    pts = list(keypoints.values())
    for a, b in connections:
        cv2.line(annotated, pts[a], pts[b], (0, 180, 255), 1)
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")
    return annotated_b64

def analyze_image(image_bytes: bytes):
    img = decode_image(image_bytes)
    h, w = img.shape[:2]
    gray, blurred = preprocess(img)
    largest, x, y, cw, ch = find_animal(img)
    body_length, height_withers, chest_width, rump_angle = extract_measurements(largest, x, y, cw, ch)
    print(f"cw={cw} ch={ch} aspect={round(cw/ch,2)} chest={chest_width} height={height_withers}")
    atc_score = calculate_score(body_length, height_withers, chest_width, rump_angle)
    bcs = calculate_bcs(img, x, y, cw, ch)
    breed = estimate_breed(img, x, y, cw, ch, body_length, height_withers, chest_width, rump_angle, bcs)
    annotated_b64 = annotate_image(img, largest, x, y, cw, ch)
    notes = (
        f"Animal detected at region ({x},{y}), "
        f"size {cw}x{ch}px in a {w}x{h} image. "
        f"Estimated {breed} based on body proportions."
    )
    return {
        "body_length":          body_length,
        "height_withers":       height_withers,
        "chest_width":          chest_width,
        "rump_angle":           rump_angle,
        "atc_score":            atc_score,
        "breed":                breed,
        "body_condition_score": bcs,
        "notes":                notes,
        "annotated_image":      annotated_b64,
    }