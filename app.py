import os
import io
import sys
import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify, render_template

# Ensure path includes current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_def import AASIST
from xai import generate_xai_heatmap

app = Flask(__name__)

# Constants
THRESHOLD = 0.0651
SAMPLE_RATE = 16000
MAX_LEN = 64600

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Initialize and load the model
model = AASIST().to(DEVICE)
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eer228.pt')

if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
    sys.exit(1)

try:
    ckpt = torch.load(weights_path, map_location=DEVICE)
    state = ckpt['model_state']
    # Clean the state dict keys (strip 'module.' prefix and ignore 'n_averaged')
    state_cleaned = {
        k.replace('module.', '', 1): v
        for k, v in state.items()
        if k != 'n_averaged'
    }
    model.load_state_dict(state_cleaned, strict=False)
    model.eval()
    print("Successfully loaded model weights from checkpoint.")
    print(f"Checkpoint info: Epoch={ckpt.get('epoch')}, Best EER={ckpt.get('best_eer', 0)*100:.2f}%")
except Exception as e:
    print(f"Error loading model weights: {e}")
    sys.exit(1)


def preprocess_audio(file_path, target_sr=SAMPLE_RATE, max_len=MAX_LEN):
    """
    Loads an audio file, resamples to target_sr, converts to mono,
    and applies paper-exact repeat padding or cropping to max_len.
    """
    # librosa.load automatically converts to mono (mono=True by default) and resamples to target_sr
    y, sr = librosa.load(file_path, sr=target_sr)
    
    # Crop or pad
    if len(y) >= max_len:
        y = y[:max_len]
    else:
        # repeat_pad
        reps = int(np.ceil(max_len / len(y)))
        y = np.tile(y, reps)[:max_len]
        
    return y


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/presentation')
def presentation():
    return render_template('presentation.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400
        
    temp_path = None
    try:
        # Create temp folder if not exists
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file to temp location to ensure librosa can read it properly
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = '.wav'
        temp_path = os.path.join(temp_dir, f"temp_upload{ext}")
        file.save(temp_path)
        
        # Preprocess audio
        y = preprocess_audio(temp_path)
        
        # Run model prediction
        with torch.no_grad():
            x_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            spoof_prob = float(probs[0, 1].cpu().item())
            bonafide_prob = float(probs[0, 0].cpu().item())
            
        verdict = "spoof" if spoof_prob > THRESHOLD else "bonafide"
        
        # Calculate confidence metric
        if verdict == "spoof":
            # Map [THRESHOLD, 1.0] to [50%, 100%]
            display_confidence = 50.0 + (spoof_prob - THRESHOLD) / (1.0 - THRESHOLD) * 50.0
            risk_level = "High" if display_confidence > 80 else "Medium"
        else:
            # Map [0, THRESHOLD] to [100%, 50%]
            display_confidence = 100.0 - (spoof_prob / THRESHOLD) * 50.0
            risk_level = "Low"
            
        display_confidence = min(max(display_confidence, 0.0), 100.0)
        
        # ─── Grad-CAM Generation ───
        heatmap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Clean up old heatmaps (older than 10 minutes)
        try:
            import time
            now = time.time()
            for f_name in os.listdir(heatmap_dir):
                fp = os.path.join(heatmap_dir, f_name)
                if os.path.isfile(fp) and now - os.path.getmtime(fp) > 600:
                    os.remove(fp)
        except Exception as ex:
            print(f"Failed to clean old heatmaps: {ex}")
            
        # Generate new heatmap
        import uuid
        image_name = f"gradcam_{uuid.uuid4().hex}.png"
        output_image_path = os.path.join(heatmap_dir, image_name)
        
        pred_class = 1 if verdict == "spoof" else 0
        generate_xai_heatmap(temp_path, model, pred_class, output_image_path)
        
        heatmap_url = f"/static/heatmaps/{image_name}"
        
        return jsonify({
            'success': True,
            'verdict': verdict,
            'spoof_prob': spoof_prob,
            'bonafide_prob': bonafide_prob,
            'threshold': THRESHOLD,
            'confidence': round(display_confidence, 2),
            'risk_level': risk_level,
            'heatmap_url': heatmap_url
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as ex:
                print(f"Failed to remove temp file: {ex}")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
