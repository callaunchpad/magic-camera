import requests
from PIL import Image
from io import BytesIO

url = 'http://52.25.237.192:8000/jimmyinator' # any mode api endpoint here

# The path to the file you want to upload
file_path = 'out/test.jpg'

# The key in the response JSON where the image URL is stored
# This is a placeholder; you'll need to replace it with the actual key
image_url_key = 'url'

# Send the POST request with the file
with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# Check if the request was successful
if response.status_code == 200:
    # Attempt to get the image URL from the response
    response_json = response.json()
    image_url = response_json.get(image_url_key)
    
    if image_url:
        # Make a request to get the image data
        image_response = requests.get(image_url)
        
        if image_response.status_code == 200:
            # Load the image data into PIL
            image = Image.open(BytesIO(image_response.content))
            image.show()  # This will display the image if possible, depending on your environment
        else:
            print(f"Failed to retrieve image: {image_response.status_code}")
    else:
        print("Image URL not found in response.")
else:
    print(f"Error with file upload: {response.status_code}, {response.text}")
