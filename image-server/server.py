from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import requests
import replicate

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from cog import BaseModel, Path

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
            "hehe": "hehe",
            "IDKWTQO": "idkwtqo",
            "serica": "serica",
            "StyleSwirl": "sannuv",
            "Eyeful": "eyeful",
            "fAIshbowlML": "fishbowl"}

# Function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def send_image_in_slack(image_url, original_path = None, msg="", mode_name = "", filetype="jpg"):
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

    team_channel_id = "C06T72ZL60J"
    launchpad_channel_id = "C06STTRLU4D"

    try:
        # Channel ID to where you want to send the message
        
        # Uploads the file
        if not original_path:
            response = client.files_upload(channels=launchpad_channel_id, file=image_path, initial_comment=msg)
            assert response["file"]  # Just checks if the file was uploaded successfully
        else:
            response = client.chat_postMessage(
                channel=launchpad_channel_id, 
                text=f"Here are the original and modified images using `{mode_name}`:"
            )


            # send images by first storing them in another channel
            original_img_response = client.files_upload(channels=team_channel_id, file=original_path)
            original_image_url = original_img_response['file']['permalink']

            modified_img_response = client.files_upload(channels=team_channel_id, file=image_path)
            modified_image_url = modified_img_response['file']['permalink']

            response = client.chat_postMessage(
                channel=launchpad_channel_id,
                text=f"the images with {mode_name}",
                blocks=[
                    {
                        "type": "image",
                        "title": {
                        "type": "plain_text",
                        "text": "Original Image"
                    },
                    "image_url": original_image_url,
                    "alt_text": "original_img"
                    },
                    {
                        "type": "image",
                        "title": {
                        "type": "plain_text",
                        "text": msg
                    },
                    "image_url": modified_image_url,
                    "alt_text": "mode_img"
                    }
                ]
            )
            print("msg sent succesfully!")
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
            "eshaanmoorjani/fishbowl:fe488f1eae23fdfa7f9082c8df788de564728a6be4b358ec9bb13f6a90857e72",
            # version 1.0 "eshaanmoorjani/fishbowl:7671f767c8b5039c0fc5e5c24c34766eb9a5ed1f4f3e5317ed0c9cc4f499eaf8",
            input=input
        )

        send_image_in_slack(output, original_path = filepath, mode_name = "fAIshbowlML", msg="Made with fAIshbowlML 🐟")

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

        send_image_in_slack(output, original_path = filepath, mode_name = "StyleSwirl", msg="Made with StyleSwirl 🍭")

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
            "eshaanmoorjani/idkwtqo:7ba0c065543f31a7785ea40d7ad5fc6fb0a85042037ec149272411ab941e874c",
            input=input
        )
        prompt = output['output_prompt']
        vid_url = output['output_vid']

        send_image_in_slack(vid_url, msg=f"Prompt: {prompt}\n\nMade with IDWTQO (ask Erica what it stands for!) 🎻", filetype="mp4")

        return {'url': output}

    return 'File type not allowed', 400

@app.route('/eyeful', methods=['POST'])
def eyeful():
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

        print("running eyeful! ")
        # deployment = replicate.deployments.get("eshaanmoorjani/serica-msd")
        # prediction = deployment.predictions.create(
        #     input=input
        # )
        # prediction.wait()
        # output=prediction.output

        output = replicate.run(
            "eshaanmoorjani/annie-goofyeyes:6d0af7acf4740769d00b7d10a5c655be38dd9e1255a615b751ffbc629956ba1a",
            input=input
        )

        send_image_in_slack(output, original_path = filepath, mode_name = "Eyeful", msg="Made with Eyeful 👀")

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

        send_image_in_slack(output, msg="Made with SERICA 🦒")

        return {'url': output}

    return 'File type not allowed', 400

@app.route('/hehe', methods=['POST'])
def hehe():
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
            "num_inference_steps": 50
        }

        print("running styleblend! ")

        output = replicate.run(
            "eshaanmoorjani/hehe:9d0ccdf81c22e98b41f2dad9abbbf83a07d4d49d577a1e755f6d0c99d5ce75c5",
            input=input
        )

        send_image_in_slack(output, original_path = filepath, mode_name = "hehe", msg="Made with hehe 🍄")

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

        send_image_in_slack(output, original_path = filepath, mode_name = "StyleBlend", msg="Made with StyleBlend 🌸")

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

        send_image_in_slack(output, original_path=filepath, mode_name = "Jimmy-inator", msg="Made with the Jimmy-inator 🤤")

        return {'url': output}
    
    return 'File type not allowed', 400

if __name__ == '__main__':
    app.run(debug=True)
