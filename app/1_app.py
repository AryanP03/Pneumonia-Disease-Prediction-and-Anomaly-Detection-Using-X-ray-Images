# --- Imports ---
import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# --- ML/Image Processing Imports --- 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import MeanSquaredError

# --- Generative AI Imports ---
import google.generativeai as genai

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, origins="http://127.0.0.1:5500", supports_credentials=True)

# --- Load Your Models ---
try:
    classifier_model = load_model('pneumonia_resnet_best_model_1.h5')
    print("--- Classification Model loaded successfully ---")
except Exception as e:
    print(f"--- Error loading classification model: {e} ---")
    classifier_model = None

# --- NEW: Load Sub-Classification Model ---
try:
    sub_classifier_model = load_model('bacteria_vs_viral_resnet_best_model_2nd_attempt.h5')
    print("--- Sub-Classification Model (Bacterial/Viral) loaded successfully ---")
except Exception as e:
    print(f"--- Error loading sub-classification model: {e} ---")
    sub_classifier_model = None

# --- Configure Generative AI ---
try:
    API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDSTTkSOCGJLhtpFbcTkgomKvdtDnJndho') # Fallback
    if 'YOUR_API_KEY' in API_KEY:
        print("--- WARNING: GOOGLE_API_KEY not found. Guidance will be disabled. ---")
        genai_model = None
    else:
        genai.configure(api_key=API_KEY)
        genai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("--- Generative AI Model configured successfully ---")
except Exception as e:
    print(f"--- Error configuring Generative AI: {e} ---")
    genai_model = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- MODIFIED: Generative AI Function for initial guidance ---
def generate_initial_guidance(prediction_text):
    if not genai_model:
        return "Guidance feature is currently unavailable."
    
    prompt = f"You are an AI health assistant. A chest X-ray analysis has just been performed with the primary finding of '{prediction_text}'. Provide an initial, empathetic, and helpful message. Explain that this is a preliminary assessment and strongly recommend consulting a medical professional. Keep it concise and start the conversation. Do not use markdown."
    try:
        response = genai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"--- Gemini API call failed: {e} ---")
        return "Could not generate initial guidance at this time."

# --- Grad-CAM XAI Functionality ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0] # For binary classification with single output
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8) # Add epsilon for stability
    return heatmap.numpy()

def generate_heatmap_image_base64(original_image_path, heatmap):
    img = cv2.imread(original_image_path)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    is_success, buffer = cv2.imencode(".png", superimposed_img)
    if not is_success: return None
    return base64.b64encode(buffer).decode('utf-8')

# --- MODIFIED: Full Analysis Function ---
def run_full_analysis(image_path):
    if not classifier_model: raise RuntimeError("Classification model not loaded.")

    print(f"--- Running Classification on {image_path} ---")
    img_cls = cv2.imread(image_path)
    if img_cls is None: raise ValueError(f"Image not found at path: {image_path}")
    img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
    img_cls = cv2.resize(img_cls, (224, 224))
    img_cls_norm = img_cls / 255.0
    img_cls_norm = np.expand_dims(img_cls_norm, axis=0)
    
    # Primary classification (Normal vs. Pneumonia)
    prediction_raw = classifier_model.predict(img_cls_norm)
    prediction_score = float(prediction_raw.flatten()[0])
    
    if prediction_score > 0.5:
        # If Pneumonia is detected, perform sub-classification
        prediction_text = "Pneumonia Detected" # Default text
        confidence = prediction_score

        if sub_classifier_model:
            print("--- Pneumonia detected, running sub-classification... ---")
            sub_prediction_raw = sub_classifier_model.predict(img_cls_norm)
            sub_prediction_score = float(sub_prediction_raw.flatten()[0])
            
            # Assuming the model was trained with Bacterial as class 0 and Viral as class 1
            if sub_prediction_score > 0.5:
                prediction_text = "Viral Pneumonia Detected"
            else:
                prediction_text = "Bacterial Pneumonia Detected"
        else:
            print("--- Sub-classification model not loaded, returning general diagnosis. ---")

    else:
        prediction_text = "Normal"
        confidence = 1 - prediction_score
    
    last_conv_layer_name = next((layer.name for layer in reversed(classifier_model.layers) if 'conv' in layer.name), None)
    if not last_conv_layer_name: raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")
    
    heatmap = make_gradcam_heatmap(img_cls_norm, classifier_model, last_conv_layer_name)
    heatmap_base64 = generate_heatmap_image_base64(image_path, heatmap)
    
    # Guidance will now use the more specific prediction text
    guidance_text = generate_initial_guidance(prediction_text)

    result = {
        "prediction": prediction_text, "confidence": round(confidence, 2),
        "guidance": guidance_text, "heatmap_image_base64": heatmap_base64
    }
    print(f"--- Initial Analysis Result Sent: {prediction_text} ---")
    return result

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files: return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            prediction_data = run_full_analysis(filepath)
            return jsonify(prediction_data)
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            return jsonify({"error": "Failed to process the image"}), 500
        finally:
            if os.path.exists(filepath): os.remove(filepath)
    else:
        return jsonify({"error": "File type not allowed"}), 400

# --- NEW: Chat Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    if not genai_model:
        return jsonify({"error": "Chat feature is not configured."}), 503
    
    data = request.get_json()
    history = data.get('history', [])
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Empty message received."}), 400

    # Reformat history for the Gemini API
    chat_session = genai_model.start_chat(history=[])
    for msg in history:
        role = 'user' if msg['sender'] == 'user' else 'model'
        chat_session.history.append({'role': role, 'parts': [{'text': msg['content']}]})
    
    try:
        print(f"--- Sending to Gemini: {user_message} ---")
        response = chat_session.send_message(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"--- Gemini API chat call failed: {e} ---")
        return jsonify({"error": "Could not get a response from the AI assistant."}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

