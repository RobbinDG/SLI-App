from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/transfer_blob')
def transfer_blob_req():
    json = request.json
    if 'blob' in json:
        blob = json['blob']


if __name__ == '__main__':
    app.run()
