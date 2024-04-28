from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import requests
import replicate

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Your bot's OAuth token
slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

app = Flask(__name__)

# Ensure there's a folder named 'uploads' in the same directory as this script.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
endpoints = {"StyleBlend": "styleblend",
            "Jimmy-Inator": "jimmyinator",
            "IDKWTQO": "idkwtqo",
            # "serica": "serica",
            "StyleSwirl": "sannuv",
            "fAIshbowlML": "fishbowl"}

# Function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_image_in_slack(image_url, filetype="jpg"):
    # Local path where you want toe the image, including the filename and extension
    image_path = f"uploads/to_slack.{filetype}"


    # Fetch the image
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file with write-binary mode and save the image to it
        with open(image_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Error downloading the image: {response.status_code}")

    for channel_id in ['C06STTRLU4D', "C06T72ZL60J"]:
        try:
            # Channel ID to where you want to send the message
            
            # Uploads the file
            response = client.files_upload(channels=channel_id, file=image_path)
            assert response["file"]  # Just checks if the file was uploaded successfully
        except SlackApiError as e:
            # You will get a SlackApiError if "ok" is False
            assert e.response["ok"] is False
            assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
            print(f"Got an error: {e.response['error']}")

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
        # deployment = replicate.deployments.get("eshaanmoorjani/fishbowl-msd")
        # prediction = deployment.predictions.create(
        #     input=input
        # )
        # prediction.wait()
        # output = prediction.output

        output = replicate.run(
            "eshaanmoorjani/fishbowl:7671f767c8b5039c0fc5e5c24c34766eb9a5ed1f4f3e5317ed0c9cc4f499eaf8",
            input=input
        )

        send_image_in_slack(output)

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
    
        # deployment = replicate.deployments.get("eshaanmoorjani/sannuv")
        # prediction = deployment.predictions.create(input=input)
        # prediction.wait()
        # output = prediction.output

        output = replicate.run(
            "eshaanmoorjani/sannuv:0fb8b58080389501f54b487b56b8023e4431a1308ce9f569b276fe84bd588cba",
            input=input
        )

        send_image_in_slack(output)

        return {'url': output}

    return 'File type not allowed', 400

@app.route('/idkwtqo', methods=['POST'])
def idkwtqo():
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

        print("running idkwtqo! ")

        output = replicate.run(
            "eshaanmoorjani/idkwtqo:8f5b2e22492b3e8074c7ee1cb9bafd835558abb7b5c004728917263dcb2cec00",
            input=input
        )

        send_image_in_slack(output, filetype="wav")

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
        # deployment = replicate.deployments.get("eshaanmoorjani/serica-msd")
        # prediction = deployment.predictions.create(
        #     input=input
        # )
        # prediction.wait()
        # output=prediction.output

        output = replicate.run(
            "eshaanmoorjani/serica:396eee9314e153aa07d261b883fb83ba97858637af7eff32502fce73eeda3509",
            input=input
        )

        send_image_in_slack(output)

        return {'url': output}

    return 'File type not allowed', 400

@app.route('/styleblend', methods=['POST'])
def styleblend():
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

        print("running styleblend! ")

        output = replicate.run(
            "eshaanmoorjani/styleblend:235c1a1787d88dab7f495f6175d0add62073d50d2f1926ea94f3c0d8ee4b5bbc",
            input=input
        )

        send_image_in_slack(output)

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

        send_image_in_slack(output)

        return {'url': output}
    
    return 'File type not allowed', 400

if __name__ == '__main__':
    app.run(debug=True)
