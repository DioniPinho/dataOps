import os

import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float

app = FastAPI(title="Fetal Health API",
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get api health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction for fetal health"
                  }
              ])

def load_model():
    print("reading model...")
    MLFLOW_TRACKING_URI = ""
    MLFLOW_TRACKING_USERNAME = ""
    MLFLOW_TRACKING_PASSWORD = ""

    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    print('setting mlflow...')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print('creating client...')
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    print('getting registered model...')
    registered_model = client.get_registered_model("fetal registry")
    registered_model.latest_versions

    print('reading model......')
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
    return loaded_model

@app.on_event("startup")
def startup_event():
    global loaded_model
    loaded_model = load_model()


@app.get("/",
         tags=["Health"])
def api_health():
    return {"status": "healthy"}


@app.post("/predict",
          tags=["Prediction"])
def api_predict(request: FetalHealthData):
    global loaded_model
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    print(received_data)
    prediction = loaded_model.predict(received_data)
    print(prediction)

    return {"prediction": str(np.argmax(prediction[0]))}
