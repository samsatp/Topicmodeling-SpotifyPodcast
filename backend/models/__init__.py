import backend.config as config
import tensorflow as tf
import numpy as np
import re

class dummy_1():
    def __init__(self):
        self.W = 10

    def predict(self, x_0:float, x_1:float):
        return self.W*x_1 + x_1 


#finetuned_loaded = tf.keras.models.load_model(config.modelPaths.finetuned_path)

class model_2():
    def __init__(self):
        self.name = "model_2"
        self.info = {
            "framework":"tensorflow",
            "version":tf.__version__
        }
        self.model = tf.keras.models.load_model(config.modelPaths.model2_path)
        self.encoder_classes = np.load(config.dataPaths.model_2_encoder_classes, allow_pickle=True)
    
    def preprocess(self, sentence:str):
        # Let sender do: sentence = re.sub('\"','', sentence)
        # Todo: sentence = re.sub("\d\d:\d\d:\d\d","[...]",sentence)
        sentence = re.sub(config.preprocessing.regex_url, "[UNK]", sentence)
        return sentence

    def get_prediction(self, sentence:str):
        sentence = self.preprocess(sentence)
        model_input = tf.constant([sentence], dtype=tf.string)
        pred = self.model.predict(model_input)
        return pred.flatten().tolist(), sentence