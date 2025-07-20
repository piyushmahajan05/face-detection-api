from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
from PIL import Image
import os
import shutil
import uuid
import glob

app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    # Step 1: Save uploaded image temporarily
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 2: Run YOLOv5 detection
    results = model(temp_filename)
    results.save()  # Saves to runs/detect/exp*

    # Step 3: Clean up temp image
    os.remove(temp_filename)

    # Step 4: Get latest saved folder path
    try:
        saved_folders = sorted(glob.glob('runs/detect/exp*'), key=os.path.getmtime)
        latest_folder = saved_folders[-1] if saved_folders else "Not Found"
    except Exception as e:
        latest_folder = f"Error getting folder: {str(e)}"

    # Step 5: Return detection results + output image path
    return JSONResponse(content={
        "message": "Detection completed",
        "detections": results.pandas().xyxy[0].to_dict(orient="records"),
        "output_folder": latest_folder
    })

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
