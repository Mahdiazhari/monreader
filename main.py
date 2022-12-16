from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import expand_dims
from tensorflow.nn import sigmoid
from tensorflow import where
from numpy import argmax
import keras.backend as K
from numpy import max
from numpy import array
from json import dumps
from uvicorn import run
import os

app = FastAPI()
model_dir = "./model_weights/monreader_mobilenetv2_finetuned.h5"


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


model = load_model(model_dir, custom_objects={"f1_metric": f1_metric})

class_predictions = ["flip", "notflip"]

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Monreader API"}


@app.post("/image/prediction/")
async def get_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}

    img_path = get_file(origin=image_link)

    img = load_img(img_path, target_size=(180, 180))

    img_array = img_to_array(img)
    img_batch = expand_dims(img_array, 0)

    pred = model.predict_on_batch(img_batch).flatten()
    score = sigmoid(pred)

    prediction_result = where(score < 0.5, 0, 1)

    class_prediction = class_predictions[prediction_result[0]]

    if prediction_result.numpy()[0] < 0.5:
        model_score = "{:.2f}".format(100 - score[0])
    else:
        model_score= "{:.2f}".format(100 * score[0])

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score,
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)
