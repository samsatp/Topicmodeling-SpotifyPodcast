import backend.config as config
import tensorflow as tf
import re

class dummy_1():
    def __init__(self):
        self.W = 10

    def predict(self, x_0:float, x_1:float):
        return self.W*x_1 + x_1 

model_2_loaded = tf.keras.models.load_model(config.modelPaths.model2_path)
#finetuned_loaded

class model_2():
    def __init__(self):
        self.name = "model_2"
        self.info = {
            "framework":"tensorflow",
            "version":tf.__version__
        }
        self.model = model_2_loaded
    
    def preprocess(self, sentence:str):
        return re.sub(config.preprocessing.regex_url, "[UNK]", sentence)

    def get_prediction(self, sentence:str):
        sentence = self.preprocess(sentence)
        model_input = tf.constant([sentence], dtype=tf.string)
        pred = self.model.predict(model_input)
        return pred.flatten()