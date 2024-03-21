from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import replicate

app = Flask(__name__)

# Ensure there's a folder named 'uploads' in the same directory as this script.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
endpoints = {"Jimmy-Inator": "jimmyinator"}

# Function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/endpoints', methods=['GET'])
def get_endpoints():
    print("getting endpoints")
    return endpoints


@app.route('/jimmyinator', methods=['POST'])
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

        input = {
            "swap_image": open("./example_jimmyinator/jimmyli.jpeg", 'rb'),
            "target_image": open(filepath, 'rb'),
        }

        print("running jimmy faceswap! ")
        output = replicate.run(
            "lucataco/faceswap:9a4298548422074c3f57258c5d544497314ae4112df80d116f0d2109e843d20d",
            input=input
        )
        return output
    
    return 'File type not allowed', 400

if __name__ == '__main__':
    app.run(debug=True)
