import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import pytesseract
import io
import re
import cv2
import numpy as np

app = FastAPI()

# Preprocess image for better OCR accuracy
def preprocess(image: Image.Image) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, sharp_kernel)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# Extract username from image (DO NOT MODIFY)
def extract_username(image: Image.Image) -> str:
    processed = preprocess(image)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=config)
    text_cleaned = text.replace('hitps', 'https').replace('httos', 'https').replace('levr', 'kvr')

    match = re.search(r'leetcode\.com/u/([a-zA-Z0-9_]+)', text_cleaned)
    return match.group(1) if match else "Unknown"

# Scrape LeetCode profile for stats
def fetch_leetcode_stats(username: str) -> dict:
    url = f"https://leetcode.com/{username}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"status": "error", "message": "Profile not found"}

    soup = BeautifulSoup(response.text, "html.parser")

    def extract_stat(pattern):
        element = soup.find(string=re.compile(pattern))
        return int(re.search(r"(\d+)", element).group(1)) if element else 0

    return {
        "username": username,
        "stats": {
            "totalSolved": extract_stat(r"Total Solved"),
            "easySolved": extract_stat(r"Easy"),
            "mediumSolved": extract_stat(r"Medium"),
            "hardSolved": extract_stat(r"Hard"),
            "ranking": extract_stat(r"Ranking"),
        }
    }

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """<!DOCTYPE html>
    <html>
    <head>
        <title>Upload LeetCode Profile Screenshot</title>
        <style>
            body { font-family: Arial; text-align: center; margin: 40px; }
            input[type=file] { margin: 20px; }
            #result { margin-top: 30px; white-space: pre-wrap; max-width: 600px; margin: auto; font-family: monospace; text-align: left; }
        </style>
    </head>
    <body>
        <h1>Upload LeetCode Profile Screenshot</h1>
        <form id="upload-form">
            <input type="file" id="file" accept="image/*" required />
            <br/>
            <button type="submit">Upload</button>
        </form>
        <div id="result"></div>
        <script>
            document.getElementById('upload-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('file');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const res = await fetch('/upload/', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('result').textContent = JSON.stringify(data, null, 2);
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        username = extract_username(image)
        stats = fetch_leetcode_stats(username)
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)