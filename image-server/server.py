from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Ensure there's a folder named 'uploads' in the same directory as this script.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/magic', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part in the request', 400
    file = request.files['file']
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Here, you can add any processing you need on the saved file
        return send_file(filepath, mimetype='image/jpeg')
    return 'File type not allowed', 400

if __name__ == '__main__':
    app.run(debug=True)
