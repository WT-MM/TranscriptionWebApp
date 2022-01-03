import os
import uuid
from flask import Flask, render_template, flash, request, redirect, jsonify
from ASR import ASR

app = Flask(__name__)

transcribe = ASR()

ttext={"w2v2" : ""}

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

    file_name = str(uuid.uuid4()) + ".mp3"
    full_file_name = os.path.join('tempfiles', file_name)
    file.save(full_file_name)
    transcript = transcribe.w2v2(full_file_name)
    os.remove(full_file_name)
    ttext['w2v2']=transcript[0]
    return render_template('index.html')

@app.route('/_words', methods=['GET'])
def words():
    return jsonify(w2v2=ttext['w2v2'])

@app.errorhandler(404)
def errorPage(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)

