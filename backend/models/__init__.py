import backend.config as config

from abc import ABC, abstractmethod
from transformers import DistilBertTokenizer
import tensorflow as tf
import numpy as np
import re
from enum import Enum

# Fine-tuned stuff
distilled_Tok = DistilBertTokenizer.from_pretrained(config.tokenizerPaths.distilled_tok_path)
loaded_fineTuned = tf.saved_model.load(config.modelPaths.finetuned_path)
infer = loaded_fineTuned.signatures["serving_default"]
fineTuned_classes = np.load(config.dataPaths.fineTuned_classes)

# Keras Sequential stuff
loaded_model_2 = tf.keras.models.load_model(config.modelPaths.model2_path)
model_2_classes = np.load(config.dataPaths.model_2_classes, allow_pickle=True)

class modelInfo(Enum):
    framework: str = "framework"
    version: str = "version"

class model(ABC):
    @abstractmethod
    def __init__(self, name, info, model, encoder_classes):
        self.name = name
        self.info = info
        self.model = model
        self.encoder_classes = encoder_classes

    @abstractmethod
    def preprocess(self):
        pass
    @abstractmethod
    def get_prediction(self):
        pass


class model_2(model):
    def __init__(self):
        super().__init__(
            name = "model_2", 
            info = {
                modelInfo.framework : "Tensorflow",
                modelInfo.version : "9-Aug"
            }, 
            model = loaded_model_2, 
            encoder_classes = model_2_classes
        )

    def preprocess(self, sentence:str):
        # Let sender do: sentence = re.sub('\"','', sentence)
        # TODO: sentence = re.sub("\d\d:\d\d:\d\d","[...]",sentence)
        sentence = re.sub(config.preprocessing.regex_url, "[UNK]", sentence)
        return sentence

    def get_prediction(self, sentence:str):
        sentence = self.preprocess(sentence)
        model_input = tf.constant([sentence], dtype=tf.string)
        pred = self.model.predict(model_input)
        return pred.flatten().tolist(), sentence

class fineTuned(model):
    def __init__(self):
        self.tokenizer = distilled_Tok
        super().__init__(
            name = "fine-tuned-distrilled-bert", 
            info = {
                modelInfo.framework : "TFBert-HuggingFace",
                modelInfo.version : "12-Aug",
                "checkpoint" : "distilbert-base-uncased"
            }, 
            model = infer, 
            encoder_classes = fineTuned_classes
        )

    def tokenize_seqs(self, seqs):
        encoded = self.tokenizer(
                        seqs,
                        padding='max_length',
                        truncation=True,
                        return_tensors='tf',
                    )
        return encoded.data

    def preprocess(self, sentence:str):
        tokenized = self.tokenize_seqs(sentence)
        return tokenized
    
    def get_prediction(self, sentence:str):
        tok = self.preprocess(sentence)
        pred = self.model(
            attention_mask = tok['attention_mask'], 
            input_ids = tok['input_ids']
        )
        return pred["outputs"].numpy().ravel().tolist(), sentence