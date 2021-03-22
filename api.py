import pandas as pd
import pickle
import json
import numpy as np
import io
from flask import Flask, render_template, request
from flask import send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import model_from_json
import glob
import os
app = Flask(__name__)


def load_model():
    model_name = 'classification_model'
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classification_model = model_from_json(loaded_model_json)
    classification_model.load_weights(model_name + '.h5')
    return classification_model


def load_tokenizer():
    Pkl_Filename = 'tokensizer.pkl'
    with open(Pkl_Filename, 'rb') as file:
        tokensizer = pickle.load(file)
    return tokensizer


def load_class_encoder():
    Pkl_Filename = 'class_encoder.pkl'
    with open(Pkl_Filename, 'rb') as file:
        class_encoder = pickle.load(file)
    return class_encoder

def delet_old_file():
    files = glob.glob("*.txt")
    for path in files:
        if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
        
@app.route('/emailclassifier', methods=['POST'])
def upload_file():
    model = load_model()
    tokensizer = load_tokenizer()
    encoder = load_class_encoder()
    if request.method == 'POST':
        try:
            model = load_model()
            f = request.files['file']
            f.save(secure_filename(f.filename))
            filename = f.filename.replace(' ', '_')
            df = pd.read_csv(filename)
            df['converse'] = df['converse'].astype(str)
            tokens = tokensizer.texts_to_matrix(df['converse'])
            converse = []
            labels = []
            output_df = pd.DataFrame()
            for i in range(0, len(tokens)):
                prediction = model.predict(np.array([tokens[i]]))
                text_labels = encoder.classes_
                predicted_label = text_labels[np.argmax(prediction[0])]
                labels.append(predicted_label)
                converse.append(df['converse'].iloc[i])
            output_df = pd.DataFrame({'converse': converse,
                                     'lables': labels})
            output_df.to_csv('Output.csv', index=False)
            return send_file('Output.csv', attachment_filename='Output.csv')
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run()
