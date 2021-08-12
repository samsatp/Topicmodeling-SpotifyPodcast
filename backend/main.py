from fastapi import FastAPI
from enum import Enum
from typing import Optional
import os
import importlib
import re
import tensorflow as tf
from pydantic import BaseModel

# custom modules
import backend.models as models
import backend.config as configs

class modelName(str, Enum):
    baseline = "baseline"
    model_2 = "model_2"
    fineTune = "fine-tuned-distrilled-bert"

class Query(BaseModel):
    sentence: str

model_classes = {
    "model_2": models.model_2()
}
    
app = FastAPI()

@app.get("/")
def root():
    welcome_text = "Welcome 👋, this is Topicmodelling API"
    return welcome_text

@app.get("/prediction/{model}")
def get_prediction(
    model:modelName,
    query:Query,
    verbose:int=0
    ):
    if model==modelName.model_2:
        pred, preprocessed_sentence = model_classes['model_2'].get_prediction(query.sentence)
    
    ## TODO: fix saving&loading fine-tune bug
    elif model==modelName.fineTune:
        pass
    ## TODO: save baseline model
    elif model==modelName.baseline:
        pass
        

    result = dict(zip(model_classes['model_2'].encoder_classes.tolist(), pred))
    if verbose>0:
        print(preprocessed_sentence)
        result['preprocessed sentence'] = preprocessed_sentence
    return result
    

