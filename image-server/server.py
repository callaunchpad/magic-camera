from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import replicate

app = Flask(__name__)

# Ensure there's a folder named 'uploads' in the same directory as this script.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
endpoints = {"Jimmy-Inator": "jimmyinator",
            "SERICA": "serica",
            "SANNUV": "sannuv",
            "Fishbowl": "fishbowl"}

# Function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/endpoints', methods=['GET'])
def get_endpoints():
    print("getting endpoints")
    return endpoints

@app.route('/fishbowl', methods=['POST'])
def fishbowl():
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
            "image": open(filepath, 'rb'),
        }

        print("running fishbowl! ")
        output = replicate.run(
            "eshaanmoorjani/fishbowl:7671f767c8b5039c0fc5e5c24c34766eb9a5ed1f4f3e5317ed0c9cc4f499eaf8",
            input=input
        )

        return {'url': output}

    return 'File type not allowed', 400

@app.route('/sannuv', methods=['POST'])
def sannuv():
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
            "image": open(filepath, 'rb'),
            "num_inference_steps": 10,
            "image_guidance_scale": 1.2
        }

        print("running sannuv! ")
        output = replicate.run(
            "eshaanmoorjani/sannuv:0fb8b58080389501f54b487b56b8023e4431a1308ce9f569b276fe84bd588cba",
            input=input
        )

        return {'url': output}

    return 'File type not allowed', 400

@app.route('/serica', methods=['POST'])
def serica():
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
            "image": open(filepath, 'rb'),    
            "scale": 1.5
        }

        print("running serica! ")
        output = replicate.run(
            "eshaanmoorjani/serica:396eee9314e153aa07d261b883fb83ba97858637af7eff32502fce73eeda3509",
            input=input
        )

        return {'url': output}

    return 'File type not allowed', 400

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

        return {'url': output}
    
    return 'File type not allowed', 400

if __name__ == '__main__':
    app.run(debug=True)
