from flask import Flask, request, jsonify, render_template_string
import librosa
import numpy as np
import tempfile
import os

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Rhythm Deck | Chord Engine</title>
    <style>
        body {
            margin:0;
            font-family: Arial, sans-serif;
            background:#0e0e0e;
            color:white;
            text-align:center;
        }
        header {
            background:black;
            padding:20px;
            font-size:24px;
            font-weight:bold;
            letter-spacing:2px;
            border-bottom:2px solid #ff3c00;
        }
        footer {
            background:black;
            padding:15px;
            position:fixed;
            bottom:0;
            width:100%;
            border-top:2px solid #ff3c00;
        }
        .container {
            margin-top:100px;
        }
        button {
            background:#ff3c00;
            border:none;
            padding:12px 25px;
            color:white;
            font-size:16px;
            cursor:pointer;
            margin-top:15px;
        }
        input {
            margin-top:20px;
        }
        #result {
            margin-top:30px;
            font-size:20px;
        }
    </style>
</head>
<body>

<header>
    RHYTHM DECK
</header>

<div class="container">
    <h2>Chord Detection Engine</h2>
    <input type="file" id="fileInput" accept=".mp3"/>
    <br>
    <button onclick="uploadFile()">Analyze</button>
    <div id="result"></div>
</div>

<footer>
    © 2026 Rhythm Deck Music
</footer>

<script>
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

    if (!fileInput.files.length) {
        alert("Please select an MP3 file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.innerHTML = "Analyzing...";

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = "Error: " + data.error;
        } else {
            resultDiv.innerHTML = "Estimated Chords: " + data.estimated_chords.join(", ");
        }
    })
    .catch(err => {
        resultDiv.innerHTML = "Something went wrong.";
    });
}
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        y, sr = librosa.load(temp.name)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

    os.unlink(temp.name)

    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    top_notes = np.argsort(chroma_mean)[-3:]
    chords = [notes[i] for i in top_notes]

    return jsonify({"estimated_chords": chords})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
