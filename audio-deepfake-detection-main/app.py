from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
import pathlib
import google.generativeai as genai  
from google.generativeai import GenerativeModel

GOOGLE_API_KEY = 'AIzaSyBLU00yEoUL9F9T4VBoqtC-7e0HPbglS4U'
genai.configure(api_key=GOOGLE_API_KEY)
summarization_model = GenerativeModel('models/gemini-1.5-flash')

# Load your trained model
model = tf.keras.models.load_model('audio_classifier.h5')

app = Flask(__name__)

def preprocess_audio(file_path):
    # Load your audio file
    data, sr = librosa.load(file_path, sr=None)
    
    # Convert to Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to log scale
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Ensure mel_spectrogram has the shape (128, n_frames)
    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max(0, 109 - mel_spectrogram.shape[1]))), mode='constant')
    mel_spectrogram = mel_spectrogram[:, :109]  # Truncate to 109 frames if needed

    # Add batch dimension and channel dimension
    mel_spectrogram = mel_spectrogram[np.newaxis, ..., np.newaxis]  # Shape (1, 128, 109, 1)
    
    return mel_spectrogram

def summarize_audio(audio_file_path):
    # Create the prompt for summarization
    prompt = "Please summarize the audio."

    # Ensure the audio file exists before reading
    if pathlib.Path(audio_file_path).exists():
        audio_data = {
            "mime_type": "audio/flac",
            "data": pathlib.Path(audio_file_path).read_bytes()
        }
        
        # Generate content with the model for summarization
        response = summarization_model.generate_content([prompt, audio_data])
        
        return response.text
    else:
        return "Error: The audio file does not exist."

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file.filename.endswith('.flac'):
        file_path = 'temp_audio.flac'
        file.save(file_path)

        # Preprocess the audio
        audio_data = preprocess_audio(file_path)
        print(f"Processed audio shape: {audio_data.shape}")  # Debugging line

        # Make predictions
        prediction = model.predict(audio_data)
        print(f"Prediction: {prediction}")  # Debugging line
        
        # Extract probabilities
        probability_fake = prediction[0][1]  # Probability for 'fake'
        probability_not_fake = prediction[0][0]  # Probability for 'not fake'
        
        # Determine result based on threshold
        result = 'not fake' if probability_not_fake >= 0.5 else 'fake'

        # Summarize the audio if the file is classified as 'not fake'
        if result == 'not fake':
            summary = summarize_audio(file_path)
        else:
            summary = "Audio classified as fake; no summary generated."
        print(summary)
        return jsonify({
            'result': result,
            'probability_not_fake': float(probability_not_fake),
            'probability_fake': float(probability_fake),
            'summary': summary
        })

    return jsonify({'error': 'File format not supported, please upload a .flac file'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
