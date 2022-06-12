from flask import Flask
from flask_bootstrap import Bootstrap5

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
bootstrap = Bootstrap5(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MiB


