# --- Imports ---
import os
import random
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- ML/Image Processing Imports --- 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# --- Generative AI Imports (Groq) ---
from groq import Groq

# --- Configuration & Environment Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Load environment variables (.env from root or app directory)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
load_dotenv(os.path.join(BASE_DIR, '.env'))

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))
if not os.path.isabs(UPLOAD_FOLDER):
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'nexathink-ai-medical-diagnostics-secure-prod-key-2026')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

cors_origins_env = os.getenv('CORS_ORIGINS', '*')
cors_origins = cors_origins_env.split(',') if cors_origins_env != '*' else '*'
CORS(app, resources={r"/*": {"origins": cors_origins}})

# --- CORS & Chrome Private Network Access (PNA) Handlers ---
@app.before_request
def handle_options_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        origin = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Allow-Private-Network'] = 'true'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

# --- Helper: Model Resolution ---
def find_model_path(model_filename):
    possible_paths = [
        os.path.join(BASE_DIR, model_filename),
        os.path.join(PROJECT_ROOT, model_filename),
        model_filename,
        os.path.join(r'D:\techrush_2025', model_filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# --- Load Classification Models ---
model_env_path = os.getenv('CLASSIFIER_MODEL_PATH', 'pneumonia_resnet_best_model_1.h5')
classifier_path = find_model_path(model_env_path)
if classifier_path:
    try:
        classifier_model = load_model(classifier_path)
        print(f"--- Classification Model loaded successfully from: {classifier_path} ---")
    except Exception as e:
        print(f"--- Error loading classification model: {e} ---")
        classifier_model = None
else:
    print(f"--- WARNING: {model_env_path} not found ---")
    classifier_model = None

sub_model_env_path = os.getenv('SUB_CLASSIFIER_MODEL_PATH', 'bacteria_vs_viral_resnet_best_model_2nd_attempt.h5')
sub_classifier_path = find_model_path(sub_model_env_path)
if sub_classifier_path:
    try:
        sub_classifier_model = load_model(sub_classifier_path)
        print(f"--- Sub-Classification Model (Bacterial/Viral) loaded successfully from: {sub_classifier_path} ---")
    except Exception as e:
        print(f"--- Error loading sub-classification model: {e} ---")
        sub_classifier_model = None
else:
    sub_classifier_model = None

# --- Load Auxiliary Backbone for Medical vs. Non-Medical Object Discrimination ---
try:
    validation_backbone = tf.keras.applications.MobileNetV2(weights='imagenet')
    print("--- Auxiliary Validation Model (MobileNetV2 ImageNet) loaded successfully ---")
except Exception as e:
    print(f"--- Notice: Could not load MobileNetV2 for validation ({e}). Using radiographic validator. ---")
    validation_backbone = None

# --- Chest Radiograph Anatomical & Morphological Validator ---
def validate_chest_xray(image_path):
    """
    Robust multi-criteria validation to ensure uploaded image is a legitimate chest radiograph (CXR).
    Distinguishes chest radiographs from color photos, documents/screenshots, non-thoracic medical scans,
    and arbitrary objects.
    Returns: (is_valid: bool, reason: str, confidence: float)
    """
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        return False, "Could not read the uploaded image file.", 0.0

    h, w = img.shape[:2]
    aspect_ratio = w / float(h)
    # Chest radiographs are portrait or roughly square (AP/PA views)
    if aspect_ratio < 0.50 or aspect_ratio > 1.70:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f}). Chest radiographs are typically portrait or square.", 0.0

    # 1. Monochromatic / Chromaticity Analysis
    # Real medical X-rays are strictly grayscale/monochromatic
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
        diff_gb = np.mean(np.abs(g.astype(float) - b.astype(float)))
        diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))
        avg_channel_diff = (diff_rg + diff_gb + diff_rb) / 3.0
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = np.mean(sat)
        high_sat_ratio = np.mean(sat > 50)
        
        if avg_channel_diff > 10.0 or mean_sat > 25.0 or high_sat_ratio > 0.06:
            return False, "The uploaded image contains color. Chest X-rays are monochromatic radiographs.", 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 2. Dynamic Range & Grayscale Intensity Distribution
    std_val = float(np.std(gray))
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dyn_range = p95 - p5
    
    # Flat / blank images
    if std_val < 22.0 or dyn_range < 70.0:
        return False, "Image lacks radiographic dynamic contrast.", 0.1

    # Documents, PDFs, code screenshots (predominantly solid white background)
    white_ratio = np.mean(gray > 240)
    if white_ratio > 0.35:
        return False, "Image contains excessive solid white areas (document or text screenshot).", 0.1

    # Extremity scans or empty dark frames
    black_ratio = np.mean(gray < 10)
    if black_ratio > 0.72:
        return False, "Image contains excessive empty black space.", 0.1

    mean_val = float(np.mean(gray))
    if mean_val < 25.0 or mean_val > 210.0:
        return False, "Image exposure is outside typical radiograph limits.", 0.1

    # 3. Bilateral Thoracic Symmetry Check
    gray_std = cv2.resize(gray, (256, 256))
    best_corr = -1.0
    for shift in range(-18, 19, 2):
        center = 128 + shift
        w_half = min(center, 256 - center)
        if w_half < 80:
            continue
        l = gray_std[:, center - w_half : center]
        r = cv2.flip(gray_std[:, center : center + w_half], 1)
        corr = np.corrcoef(l.flatten(), r.flatten())[0, 1]
        if corr > best_corr:
            best_corr = corr

    if best_corr < 0.65:
        return False, f"Lacks thoracic bilateral anatomical symmetry (correlation {best_corr:.2f}).", 0.2

    # 4. Mid-thoracic pulmonary profile
    mid_thorax = gray_std[64:192, :]
    horiz_profile = np.mean(mid_thorax, axis=0)
    left_lung_val = float(np.mean(horiz_profile[35:90]))
    right_lung_val = float(np.mean(horiz_profile[166:220]))
    avg_lung = 0.5 * (left_lung_val + right_lung_val)
    lung_diff_ratio = abs(left_lung_val - right_lung_val) / (avg_lung + 1e-5)
    if lung_diff_ratio > 0.65:
        return False, "Extreme lung field asymmetry.", 0.2

    # 5. Radiographic texture / internal rib edge density
    edges = cv2.Canny(gray_std, 40, 120)
    edge_density = float(np.mean(edges > 0))
    if edge_density < 0.015:
        return False, "Lacks internal radiographic rib and lung textures.", 0.2

    # 6. Deep ImageNet Object Check
    if validation_backbone is not None:
        try:
            resized_rgb = cv2.resize(rgb_img, (224, 224))
            x = tf.keras.applications.mobilenet_v2.preprocess_input(resized_rgb.astype(np.float32))
            preds = validation_backbone.predict(np.expand_dims(x, 0), verbose=0)
            top_pred = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
            top_class_name, top_conf = top_pred[1], float(top_pred[2])
            if top_conf > 0.45:
                return False, f"Identified non-medical object: {top_class_name} ({top_conf:.2f}).", 0.0
        except Exception as e:
            print(f"--- MobileNet validation warning: {e} ---")

    conf = min(0.99, max(0.70, best_corr * 0.5 + (std_val / 100.0) * 0.3 + 0.2))
    return True, "Valid chest radiograph confirmed.", round(float(conf), 2)


# --- Configure Generative AI (Groq) ---
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b').strip()
groq_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        masked_key = GROQ_API_KEY[:7] + "..." + GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 12 else "***"
        print(f"--- Groq Generative AI configured successfully (Model: {GROQ_MODEL}, Key: {masked_key}) ---")
    except Exception as e:
        print(f"--- Error configuring Groq client: {e} ---")
        groq_client = None
else:
    print("--- Notice: GROQ_API_KEY not configured. Rule-based clinical knowledge engine will be used. ---")

# --- Clinical Guidance Templates (Fallback & Baseline) ---
FALLBACK_GUIDANCE = {
    "Normal": (
        "Preliminary AI screening indicates clear lung fields without evidence of focal consolidation, "
        "pleural effusion, or acute pneumonic opacities. Please note that this is an automated screening tool. "
        "If you are experiencing persistent respiratory symptoms (cough, shortness of breath, or fever), "
        "we recommend consulting your healthcare provider for clinical evaluation."
    ),
    "Bacterial Pneumonia Detected": (
        "Preliminary AI analysis indicates radiographic patterns consistent with bacterial pneumonia "
        "(focal lobar or segmental consolidation). Bacterial pneumonia is typically caused by bacteria such "
        "as Streptococcus pneumoniae and usually responds well to prompt physician-directed antibiotic treatment. "
        "We strongly advise consulting a healthcare professional promptly for physical auscultation and prescription."
    ),
    "Viral Pneumonia Detected": (
        "Preliminary AI analysis indicates radiographic opacities consistent with viral pneumonia "
        "(diffuse bilateral interstitial infiltrates). Management typically emphasizes supportive care, "
        "hydration, rest, and antipyretics under medical supervision. If you experience worsening breathlessness "
        "or low oxygen levels, seek urgent medical care."
    ),
    "Pneumonia Detected": (
        "Preliminary AI assessment detected radiographic features suggestive of pneumonia. Pneumonia involves "
        "inflammation in air sacs in the lungs and requires formal medical diagnosis. We strongly advise "
        "scheduling an evaluation with a qualified physician or pulmonologist promptly."
    )
}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_initial_guidance(prediction_text):
    if groq_client:
        prompt = (
            f"You are NexaThink AI health assistant. A chest X-ray radiograph analysis has just been performed "
            f"with the primary finding of '{prediction_text}'. Provide a compassionate, clear, and reassuring initial message. "
            f"Explain that this is an AI preliminary screening and strongly advise discussing the result with a certified physician. "
            f"Keep it concise (2-3 sentences) without any markdown headers or asterisks."
        )
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional, empathetic clinical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            reply = completion.choices[0].message.content
            if reply and reply.strip():
                return reply.strip()
        except Exception as e:
            print(f"--- Groq guidance API call failed: {e}. Using clinical guidance fallback. ---")
    
    return FALLBACK_GUIDANCE.get(prediction_text, FALLBACK_GUIDANCE["Pneumonia Detected"])

def generate_chat_reply(user_message, history):
    if groq_client:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are NexaThink AI Medical Assistant, an empathetic, supportive, and knowledgeable clinical AI companion. "
                        "You help users understand chest X-ray findings, pneumonia types (bacterial vs viral), common symptoms, "
                        "precautionary measures, and warning signs that require emergency attention. Always maintain a professional, "
                        "reassuring tone, and remind the user that AI guidance cannot replace an in-person clinical diagnosis."
                    )
                }
            ]
            for msg in history:
                role = "user" if msg.get('sender') == 'user' else "assistant"
                content = msg.get('content', '')
                if content:
                    messages.append({"role": role, "content": content})
            
            messages.append({"role": "user", "content": user_message})

            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            reply = completion.choices[0].message.content
            if reply and reply.strip():
                return reply.strip()
        except Exception as e:
            print(f"--- Groq chat API call failed: {e}. Using clinical fallback engine. ---")

    # Smart clinical conversational fallback
    msg = user_message.lower()
    if any(k in msg for k in ['symptom', 'sign', 'feel']):
        return (
            "Common symptoms of pneumonia include a persistent cough (often producing greenish, yellow, or bloody mucus), "
            "fever, sweating, shaking chills, shortness of breath, sharp chest pain when breathing or coughing, and fatigue. "
            "If breathing difficulty becomes severe or oxygen levels drop, seek emergency care immediately."
        )
    elif any(k in msg for k in ['treatment', 'cure', 'medicine', 'antibiotic', 'medication']):
        return (
            "Treatment depends on the type of pneumonia. Bacterial pneumonia is typically treated with targeted prescription "
            "antibiotics. Viral pneumonia does not respond to antibiotics, and treatment focuses on supportive care, "
            "hydration, fever reducers, and rest. Always consult a physician for a prescription tailored to your condition."
        )
    elif any(k in msg for k in ['contagious', 'spread', 'catch']):
        return (
            "Many pathogens that cause pneumonia (both viral and bacterial) are contagious and spread through airborne "
            "droplets when an infected person coughs or sneezes. Good respiratory hygiene, handwashing, and wearing a mask "
            "help prevent transmission."
        )
    elif any(k in msg for k in ['prevent', 'prevention', 'vaccine']):
        return (
            "Key prevention strategies include receiving recommended vaccines (pneumococcal vaccine and annual influenza vaccine), "
            "practicing regular hand hygiene, avoiding smoking, and maintaining a robust immune system with proper nutrition and rest."
        )
    elif any(k in msg for k in ['doctor', 'hospital', 'emergency', 'urgent', 'when to see']):
        return (
            "You should see a doctor promptly if you suspect pneumonia. Seek emergency medical attention immediately if you "
            "notice: severe difficulty breathing, persistent high fever, chest pain that worsens when breathing, confusion, "
            "or bluish lips/fingertips."
        )
    elif any(k in msg for k in ['hello', 'hi', 'hey', 'greetings']):
        return (
            "Hello! I am your AI Medical Assistant. I can assist with questions regarding your chest X-ray screening, "
            "pneumonia symptoms, treatment protocols, and precautions. How can I help you today?"
        )
    else:
        return (
            "Thank you for reaching out. While I can provide general medical information regarding pneumonia and chest X-ray "
            "screenings, your findings should always be reviewed by a certified physician who can perform physical auscultation "
            "and order confirmatory clinical tests."
        )

# --- Grad-CAM XAI Functionality ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, is_pneumonia=True):
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # Class channel: for binary classification with sigmoid output (0: Normal, 1: Pneumonia)
        if is_pneumonia:
            class_channel = preds[0]
        else:
            class_channel = 1.0 - preds[0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_heatmap_image_base64(original_image_path, heatmap):
    img = cv2.imread(original_image_path)
    if img is None or heatmap is None:
        return None
    h, w = img.shape[:2]
    # Smooth cubic interpolation to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.maximum(heatmap_resized, 0)
    max_val = np.max(heatmap_resized)
    if max_val > 0:
        heatmap_resized = heatmap_resized / max_val
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    # Apply JET colormap: highlights pneumonia infiltrates and affected lung regions
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.65, heatmap_color, 0.45, 0)
    is_success, buffer = cv2.imencode(".png", superimposed_img)
    if not is_success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

# --- Full Analysis Pipeline ---
def run_full_analysis(image_path):
    if not classifier_model:
        raise RuntimeError("Classification model is not loaded. Please ensure pneumonia_resnet_best_model_1.h5 is present.")

    print(f"--- Running Classification on {image_path} ---")
    img_cls = cv2.imread(image_path)
    if img_cls is None:
        raise ValueError(f"Image not found or unreadable at path: {image_path}")
    img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
    img_cls = cv2.resize(img_cls, (224, 224))
    img_cls_norm = img_cls / 255.0
    img_cls_norm = np.expand_dims(img_cls_norm, axis=0)
    
    # Primary classification (Normal vs. Pneumonia)
    prediction_raw = classifier_model.predict(img_cls_norm)
    prediction_score = float(prediction_raw.flatten()[0])
    
    is_pneumonia = prediction_score > 0.5
    if is_pneumonia:
        prediction_text = "Pneumonia Detected"
        confidence = prediction_score

        if sub_classifier_model:
            print("--- Pneumonia detected, running sub-classification... ---")
            try:
                sub_prediction_raw = sub_classifier_model.predict(img_cls_norm)
                sub_prediction_score = float(sub_prediction_raw.flatten()[0])
                if sub_prediction_score > 0.5:
                    prediction_text = "Viral Pneumonia Detected"
                else:
                    prediction_text = "Bacterial Pneumonia Detected"
            except Exception as e:
                print(f"--- Sub-classification error: {e} ---")
        else:
            print("--- Sub-classification model not loaded, returning general diagnosis. ---")
    else:
        prediction_text = "Normal"
        confidence = 1.0 - prediction_score
    
    # Grad-CAM heatmap generation specifically highlighting affected regions
    heatmap_base64 = None
    try:
        last_conv_layer_name = next((layer.name for layer in reversed(classifier_model.layers) if 'conv' in layer.name), None)
        if last_conv_layer_name:
            heatmap = make_gradcam_heatmap(img_cls_norm, classifier_model, last_conv_layer_name, is_pneumonia=is_pneumonia)
            heatmap_base64 = generate_heatmap_image_base64(image_path, heatmap)
    except Exception as e:
        print(f"--- Grad-CAM generation warning: {e} ---")
    
    # Guidance
    guidance_text = generate_initial_guidance(prediction_text)

    result = {
        "prediction": prediction_text,
        "confidence": round(confidence, 2),
        "guidance": guidance_text,
        "heatmap_image_base64": heatmap_base64
    }
    print(f"--- Analysis Result: {prediction_text} (Confidence: {round(confidence, 2)}) ---")
    return result

# --- API Endpoints ---
@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, '1_index.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    return "1_index.html not found", 404

@app.route('/<path:filename>')
def serve_static(filename):
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(BASE_DIR, filename)
    return jsonify({"error": "File not found"}), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": classifier_model is not None,
        "sub_model_loaded": sub_classifier_model is not None,
        "groq_configured": groq_client is not None,
        "groq_model": GROQ_MODEL
    })

@app.route('/samples/<path:filename>')
def samples(filename):
    samples_dir = os.path.join(PROJECT_ROOT, 'data', 'samples')
    if os.path.exists(os.path.join(samples_dir, filename)):
        return send_from_directory(samples_dir, filename)
    return jsonify({"error": "Sample not found"}), 404

@app.route('/validate_xray', methods=['POST'])
def validate_xray_endpoint():
    if 'image' not in request.files:
        return jsonify({"is_valid": False, "error": "No image file provided", "message": "Please upload a valid chest X-ray image."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"is_valid": False, "error": "No selected file", "message": "Please upload a valid chest X-ray image."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"val_{filename}")
        file.save(filepath)
        try:
            is_valid, reason, conf = validate_chest_xray(filepath)
            return jsonify({
                "is_valid": is_valid,
                "confidence": conf,
                "reason": reason,
                "message": "Valid chest radiograph confirmed." if is_valid else "Please upload a valid chest X-ray image."
            })
        except Exception as e:
            print(f"Validation endpoint error: {e}")
            return jsonify({"is_valid": False, "error": str(e), "message": "Please upload a valid chest X-ray image."}), 500
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
    else:
        return jsonify({"is_valid": False, "error": "File type not allowed", "message": "Please upload a valid chest X-ray image."}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            # Guardrail: Ensure the uploaded image is a valid chest radiograph
            is_valid, validation_msg, conf = validate_chest_xray(filepath)
            if not is_valid:
                print(f"--- Predictive analysis blocked: non-chest X-ray detected ({validation_msg}) ---")
                return jsonify({
                    "error": "Please upload a valid chest X-ray image.",
                    "invalid_xray": True,
                    "details": validation_msg
                }), 400

            prediction_data = run_full_analysis(filepath)
            return jsonify(prediction_data)
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
    else:
        return jsonify({"error": "File type not allowed. Please upload PNG, JPG, or JPEG."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get('history', [])
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"error": "Empty message received."}), 400

    try:
        reply = generate_chat_reply(user_message, history)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"--- Chat processing error: {e} ---")
        return jsonify({"reply": "I am here to help. Could you please rephrase your medical question?"})

# --- Main Execution Block ---
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    print(f"--- Starting Pneumonia Prediction Web Server on http://{host}:{port} (Debug: {debug}) ---")
    app.run(host=host, port=port, debug=debug)
