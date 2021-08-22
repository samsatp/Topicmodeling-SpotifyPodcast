from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel

# custom modules
import backend.models as models
import backend.config as configs

class modelName(str, Enum):
    baseline = "baseline"
    model_2 = "model_2"
    fineTune = "fineTune"

model_classes = {
    modelName.model_2: models.model_2(),
    modelName.fineTune: models.fineTuned()
}

class Query(BaseModel):
    # Request Body
    sentence: str
    
app = FastAPI()

@app.get("/topicmodelling")
def root():
    welcome_text = "Welcome 👋, this is Topicmodelling API"
    return welcome_text

@app.get("/topicmodelling/prediction/{model}")
def get_prediction(
    model:modelName,
    query:Query,
    verbose:int=0
    ):
    if model in modelName.__members__:
        used_model = model_classes[model]
        pred, preprocessed_sentence = used_model.get_prediction(query.sentence)

    result = dict(zip(used_model.encoder_classes.tolist(), pred))
    if verbose>0:
        print(preprocessed_sentence)
        result['preprocessed sentence'] = preprocessed_sentence
    return result
    

