# UNTESTED CODE!!!
import subprocess
import time
import requests
import os
import concurrent.futures


BASE_URL = "http://52.25.237.192:8000/"

def check_wifi():
    try:
        # On Linux or MacOS, you might use 'ping -c 1 google.com'
        subprocess.check_output(['ping', '-c', '1', 'google.com'])
        return True
    except subprocess.CalledProcessError:
        return False

def perform_api_requests():
    # look at every image in `./to_process` and split it by `_`
    for to_process_filepath in os.listdir('to_process'):
        if to_process_filepath.endswith('.png'):
            time_taken, mode_endpoint_name, _ = to_process_filepath.split('_')
            print(f"processing image with mode {mode_endpoint_name}")
            # send image to API
            with open(to_process_filepath, "rb") as f:
                files = {"file": f}
                # make this multithreaded so you can do multiple requests at once
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future = executor.submit(requests.post, BASE_URL + mode_endpoint_name, files=files)
                    response = future.result()


def main():
    while True:
        if check_wifi():
            print("WiFi is connected, performing API requests...")
            perform_api_requests()
        else:
            print("WiFi not connected, retrying in 120 seconds...")
        time.sleep(120)  # Check every 120 seconds

if __name__ == '__main__':
    main()
