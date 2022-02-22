import os
import uuid
from flask import Flask, render_template, flash, request, redirect, jsonify
from ASR import ASR
import base64

import threading

app = Flask(__name__)

transcribe = ASR()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/save-record', methods=['POST'])
def save_record():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    ttext={"w2v2" : "", "sb":""}
    
    file = request.files['file']

    file_name = str(uuid.uuid4()) + ".wav"
    full_file_name = os.path.join('tempfiles', file_name)

    with open(full_file_name, 'wb') as f_vid:
        f_vid.write(file.read())
    
    # w2v2text = transcribe.w2v2(full_file_name)
    # sbtext = transcribe.sb(full_file_name)
    # esptext = transcribe.espnet(full_file_name)
    espOut, sbOut, w2v2Out = transcribe.runAll(full_file_name)
    os.remove(full_file_name)
    return jsonify(w2v2=w2v2Out[0], wtime=prettifyTime(w2v2Out[1]), sb=sbOut[0], sbtime=prettifyTime(sbOut[1]), espnet=espOut[0], esptime=prettifyTime(espOut[1]))


@app.errorhandler(404) 
def errorPage(e):
    return render_template('404.html'), 404


def prettifyTime(time):
    return str(round(time, 3))

if __name__ == '__main__':
    app.run(debug=True)
