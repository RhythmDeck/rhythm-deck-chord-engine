from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tempfile
import os

app = Flask(__name__)

# Strong CORS configuration (handles preflight properly)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Rhythm Deck Chord Engine Running"}), 200


# ===============================
# FULL ENGINE (Original Quality)
# ===============================
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_full():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)

            y, sr = librosa.load(temp.name, mono=True)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

        os.unlink(temp.name)

        notes = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]

        top_notes = np.argsort(chroma_mean)[-3:]
        chords = [notes[i] for i in top_notes]

        return jsonify({"estimated_chords": chords})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# LIGHT MODE (Render Free Memory Safe Mode)
# ==========================================
@app.route("/analyze-lite", methods=["POST", "OPTIONS"])
def analyze_lite():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)

            # MEMORY SAFE SETTINGS
            y, sr = librosa.load(
                temp.name,
                sr=16000,      # Lower sample rate
                mono=True,     # Force mono
                duration=60    # Limit to first 60 seconds
            )

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

        os.unlink(temp.name)

        notes = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]

        top_notes = np.argsort(chroma_mean)[-3:]
        chords = [notes[i] for i in top_notes]

        return jsonify({"estimated_chords": chords})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
