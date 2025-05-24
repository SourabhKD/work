from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('ai_detection_model.keras')

with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

MAX_LENGTH = 100

def predict_text(text):
    """Predict if text is AI-generated and return confidence score"""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    label = "AI-generated" if prediction > 0.5 else "Human-written"
    confidence = round(float(prediction if label == "AI-generated" else 1 - prediction) * 100, 2)
    
    return {
        'prediction': label,
        'confidence': confidence,
        'ai_prob': round(float(prediction) * 100, 2),
        'human_prob': round((1 - float(prediction)) * 100, 2),
        'model_type': 'CNN'
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    text = ""
    error = None
    
    if request.method == 'POST':
        text = request.form.get('text', '')
        word_count = len(text.split())
        
        if word_count < 50:
            error = "Text too short. Please enter at least 50 words."
        else:
            result = predict_text(text)
            result['word_count'] = word_count
    
    return render_template('index.html', result=result, text=text, error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if len(text.split()) < 50:
        return jsonify({
            'error': 'Text too short. Minimum 50 words required.'
        }), 400
    
    result = predict_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)