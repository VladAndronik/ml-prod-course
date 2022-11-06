from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel
from PIL import Image
import io

from serving.predictor import Predictor


class Input(BaseModel):
    image_file: UploadFile = File(...)


class Result(BaseModel):
    output: int


app = FastAPI()
model = Predictor.default_from_model_registry()
exts = ['jpg', 'jpeg', 'png']


@app.get('/check')
def check():
    return "Okay"


@app.post('/predict', response_model=Result)
def predict(image_file: UploadFile = File(...)) -> Result:
    if not any(image_file.filename.endswith(ext) for ext in exts):
        raise HTTPException(status_code=400, detail=f'Only images files with {exts} extensions can be uploaded.')
    request_object_content = image_file.file.read()
    image = Image.open(io.BytesIO(request_object_content))
    output = model.predict(image)
    return Result(output=output)
