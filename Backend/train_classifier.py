import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle #saves trained model so that we can use it later without retraining
import os
from pathlib import Path

DATASET_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\AD BACKEND\animal-atc-backend\Indian_bovine_breeds\Indian_bovine_breeds"

def extract_image_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # resize to standard size so all images are comparable
    img = cv2.resize(img, (128, 128))

    features = []

    # 1. COLOR HISTOGRAM — distribution of colors in the image
    # different breeds have different coat colors
    # e.g. Sahiwal is brown, Holstein is black and white, Gir is grey
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # 2. HSV COLOR HISTOGRAM — better for coat color analysis
    # HSV separates color from brightness so coat color works under different lighting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # 3. TEXTURE — how rough or smooth the coat is
    # coat texture differs between breeds
    # Sobel edge detection measures how much fine detail (texture) is in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    hist, _ = np.histogram(magnitude, bins=32, range=(0, 255))
    hist = hist / hist.sum()
    features.extend(hist)

    # 4. SHAPE — aspect ratio and contour features
    # different breeds have different body proportions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        aspect = cw / ch if ch > 0 else 1.0
        area_ratio = cv2.contourArea(largest) / (128 * 128)
        perimeter = cv2.arcLength(largest, True)
        features.extend([aspect, area_ratio, perimeter / 1000])
    else:
        features.extend([1.0, 0.5, 0.5])

    # total features per image = ~300 instead of 6
    # much more information for the model to work with
    return np.array(features, dtype=np.float32)


def load_dataset():
    dataset_path = Path(DATASET_PATH)
    breeds = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(breeds)} breeds")

    X, y = [], []
    for breed_folder in breeds:
        breed_name = breed_folder.name.replace("_", " ")
        images = list(breed_folder.glob("*.jpg")) + \
                 list(breed_folder.glob("*.jpeg")) + \
                 list(breed_folder.glob("*.png"))

        print(f"Processing {breed_name}: {len(images)} images")
        for img_path in images:
            features = extract_image_features(img_path)
            if features is not None:
                X.append(features)
                y.append(breed_name)

    return np.array(X), np.array(y)


print("Loading dataset and extracting features...")
X, y = load_dataset()
print(f"\nTotal samples: {len(X)}, Features per image: {X.shape[1]}")

# LabelEncoder converts breed name strings to numbers
# e.g. "Gir"=0, "Sahiwal"=1, "Murrah"=2 etc.
# ML models work with numbers not strings
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 80% training — model learns from this
# 20% testing — hidden from model, used to check accuracy
# stratify= makes sure each breed is represented equally in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTraining Random Forest...")

# RandomForest builds 300 decision trees
# Each tree asks a series of yes/no questions about the features and votes for a breed
# One tree can memorize training data and fail on new data
# 300 trees each trained on slightly different random subsets — their average is much more reliable
# This is called ensemble learning
# n_jobs=-1 = use all CPU cores to train faster
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.1f}%")
print(classification_report(y_test, predictions, target_names=le.classes_))

# pickle serializes the trained model into a binary file
# without this we would have to retrain every time the backend restarts
# saving it means the backend just loads the file in milliseconds
with open("breed_model.pkl", "wb") as f:
    pickle.dump(model, f)

# save label encoder too so we can convert numbers back to breed names
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel saved to breed_model.pkl")
print("Label encoder saved to label_encoder.pkl")