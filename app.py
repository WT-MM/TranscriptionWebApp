import os
import uuid
from flask import Flask, render_template, flash, request, redirect
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    file_name = str(uuid.uuid4()) + ".mp3"
    full_file_name = os.path.join('tempfiles', file_name)
    file.save(full_file_name)

    audio, rate = librosa.load(full_file_name, sr=16000)
    os.remove(full_file_name)
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
            
    # retrieve logits
    logits = model(input_values).logits
            
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)


    return render_template('index.html')


@app.errorhandler(404)
def errorPage(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)

