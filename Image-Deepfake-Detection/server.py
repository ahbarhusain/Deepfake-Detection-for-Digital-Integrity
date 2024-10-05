from flask import Flask, render_template, request, redirect
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from werkzeug.utils import secure_filename
from model import MesoNet
import google.generativeai as genai  

# API key setup for Google generative AI
GOOGLE_API_KEY = 'AIzaSyBLU00yEoUL9F9T4VBoqtC-7e0HPbglS4U'
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load MesoNet model for deepfake prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MesoNet().to(device)
model.load_state_dict(torch.load("mesonet_model.pth", map_location=torch.device('cpu')))

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = outputs[0].cpu().detach().numpy()
        return probs

@app.route('/')
def home():
    return redirect('/predict')

@app.route('/predict', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html')

        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html')
        if file:
            # Save the uploaded image to a folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            prob = predict(file_path)
            prob = prob[0]
            prob_real=prob
            prob_fake=1-prob_real
            if prob > 0.5:
                prediction_label = 'Real'
                #prob_real = 1.0
                #prob_fake = 0.0
            else:
                prediction_label = 'Fake'
                #prob_fake = 1.0
                #prob_real = 0.0

            # Generate recommendations using Google Gemini
            deepfake_prob = prob_fake
            img = Image.open(file_path)

            # Generate a response with probability information using Google Generative AI
            model_gemini = genai.GenerativeModel(model_name="gemini-1.5-pro")
            prompt = f"What does the image depict"
            #prompt = f"The accurate trained deepfake detection model has assigned a probability of {deepfake_prob*100}% that this image is fake. Based on the model's analysis, explain why this conclusion could have reached. Include specific features or patterns in the image that could indicate manipulation."
            #print(prompt)
            if prob_real>0.5:
                response = model_gemini.generate_content([prompt, img])
            else:
                response="Image classified as fake; no summary generated."
            #response = model_gemini.generate_content([prompt, img])

            # Plotting the results in a bar chart
            labels = ['Real', 'Fake']
            plt.figure(figsize=(4, 3))
            plt.bar(labels, [prob_real * 100, prob_fake * 100], color=['blue', 'red'], width=0.5)
            plt.ylabel('Percentage (%)')
            plt.title('Prediction Result')
            plt.ylim(0, 100)  
            plt.yticks([0, 20, 40, 60, 80, 100])

            # Save the plot to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graph_string = base64.b64encode(buffer.getvalue()).decode()

            # Return prediction result, image analysis from Gemini, and graph
            return render_template('predict.html', graph=graph_string, prediction=prediction_label, filename=filename, gemini_response=response)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=False, port=3000)
