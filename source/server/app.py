import os

from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from logmmse import logmmse
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import noisereduce as nr

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/transfer_ogg', methods=['POST'])
def transfer_blob_req():
    files = request.files
    if 'oggfile' in files:
        files['oggfile'].save('tmp.ogg')
        audio = AudioSegment.from_file('tmp.ogg')
        samples = audio.get_array_of_samples()
        for i in range(2, len(samples) - 2):
            samples[i] = int((samples[i - 2] + 2 * samples[i - 1] + 4 * samples[i] + 2 * samples[
                i + 1] + samples[i + 2]) / 10)
        new_audio = audio._spawn(data=samples)
        new_audio.export('audio.mp3', format='mp3')
    return jsonify()


if __name__ == '__main__':
    app.run()
