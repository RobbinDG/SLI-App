from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from pydub import AudioSegment
import subprocess

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
        new_audio.export('tmp.mp3', format='mp3')

        args = (
            '../classifier/spp', '../classifier/params/ser.pt', '2',
            'tmp.mp3')
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = str(popen.stdout.read())
        output = output.split('\\n')
        return jsonify(language=list(float(x) for x in output[2].split(' ')[0:-1]))
    return jsonify(language=-1)


if __name__ == '__main__':
    app.run()
