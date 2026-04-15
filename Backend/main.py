from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from classifier import analyze_image
import sqlite3
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    conn = sqlite3.connect("atc_records.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_id TEXT,
            timestamp TEXT,
            breed TEXT,
            atc_score INTEGER,
            body_length REAL,
            height_withers REAL,
            chest_width REAL,
            rump_angle REAL,
            bcs REAL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.get("/")
def home():
    return {"message": "ATC Backend running"}

@app.post("/analyze-animal")
async def analyze_animal(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_data = await file.read()
    try:
        result = analyze_image(image_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-record")
async def save_record(data: dict):
    conn = sqlite3.connect("atc_records.db")
    conn.execute("""
        INSERT INTO records
        (animal_id, timestamp, breed, atc_score, body_length, height_withers, chest_width, rump_angle, bcs, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("animal_id"),
        datetime.now().isoformat(),
        data.get("breed"),
        data.get("atc_score"),
        data.get("body_length"),
        data.get("height_withers"),
        data.get("chest_width"),
        data.get("rump_angle"),
        data.get("body_condition_score"),
        data.get("notes"),
    ))
    conn.commit()
    conn.close()
    return {"status": "saved"}

@app.get("/records")
async def get_records():
    conn = sqlite3.connect("atc_records.db")
    rows = conn.execute("SELECT * FROM records ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [
        {
            "id": r[0], "animal_id": r[1], "timestamp": r[2],
            "breed": r[3], "atc_score": r[4], "body_length": r[5],
            "height_withers": r[6], "chest_width": r[7],
            "rump_angle": r[8], "bcs": r[9], "notes": r[10],
        }
        for r in rows
    ]
    
