import os
import uuid
from flask import Flask, render_template, flash, request, redirect, jsonify
from ASR import ASR
import base64

import librosa

app = Flask(__name__)

transcribe = ASR()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    
    ttext={"w2v2" : "", "sb":""}
    
    file = request.files['file']

    file_name = str(uuid.uuid4()) + ".wav"
    full_file_name = os.path.join('tempfiles', file_name)

    with open(full_file_name, 'wb') as f_vid:
        f_vid.write(file.read())
    
    w2v2text = transcribe.w2v2(full_file_name)
    sbtext=transcribe.sb(full_file_name)

    os.remove(full_file_name)
    return jsonify(w2v2=w2v2text[0],sb=sbtext)


@app.errorhandler(404) 
def errorPage(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)

