from fastapi import FastAPI, File, UploadFile, HTTPException
from celery import Celery
import os
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
from pydantic import BaseModel
from typing import List

app = FastAPI()

celery = Celery('carsounds-sm', broker='redis://localhost:6379/0')
celery.conf.update({
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0'
})

@celery.task
def analyze_file(file_path):
    try:
        c, p, p_nam = aT.file_classification(file_path, "motorsoundsmodel", "gradientboosting")

        n = np.array(p)
        maxindex = np.argmax(n)
        predicted_class = p_nam[maxindex]
        confidence = round(max(p), 5)

        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_sounds(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    task_ids = []
    for file in files:
        file_path = os.path.join('/tmp', file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        task = analyze_file.apply_async(args=[file_path])
        task_ids.append(task.id)

    return {"task_ids": task_ids, "status": "Tasks started"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = analyze_file.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {
            'status': task.state,
            'result': 'Task is still pending'
        }
    elif task.state != 'FAILURE':
        return {
            'status': task.state,
            'result': task.result
        }
    else:
        return {
            'status': task.state,
            'result': str(task.info),
        }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)