import mimetypes
import json
import os
import time
import subprocess

import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
from firebase_admin import credentials, initialize_app, storage, firestore

import torch
from diffusers import PixArtAlphaPipeline
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils.rp_validator import validate

from mora import load_models, Get_image, generate_and_concatenate_videos

logger = RunPodLogger()

LOCAL_URL = "http://127.0.0.1:5000"

SERVICE_CERT = json.loads(os.environ["FIREBASE_KEY"])
# SADTALKER_SERVICE_CERT = json.loads(os.environ["SADTALKER_FIREBASE_KEY"])
STORAGE_BUCKET = os.environ["STORAGE_BUCKET"]

cred_obj = credentials.Certificate(SERVICE_CERT)
# sad_cred_obj = credentials.Certificate(SADTALKER_SERVICE_CERT)

default_app = initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET}, name='mora')
# sad_app = initialize_app(sad_cred_obj, name='sadtalker')


INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "title": "Prompt",
        "default": "A boy",
        "required": True,
        "description": "Prompt for video generation."
    }
}


def get_extension_from_mime(mime_type):
    extension = mimetypes.guess_extension(mime_type)
    return extension


def upload_video(filename):
    destination_blob_name = f'mora/{filename}'
    bucket = storage.bucket(app=default_app)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(filename)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    logger.info("File uploaded to firebase...")
    return blob.public_url


# cog_session = requests.Session()
# retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
# cog_session.mount('http://', HTTPAdapter(max_retries=retries))


# ----------------------------- Start API Service ---------------------------- #
# Call "python -m cog.server.http" in a subprocess to start the API service.
# subprocess.Popen(["python", "-m", "cog.server.http"])


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            health = requests.get(url, timeout=120)
            status = health.json()["status"]

            if status == "READY":
                time.sleep(1)
                return

        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    prompt = inference_request["prompt"]
    pipe, base, refiner, n_steps, high_noise_frac = load_models()

    # Example usage
    # Get_image("A boy", base, refiner, n_steps, high_noise_frac)
    Get_image(prompt, base, refiner, n_steps, high_noise_frac)
    output_file = generate_and_concatenate_videos("image.png", pipe, 3)
    return output_file


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    validated_input = validate(event["input"], INPUT_SCHEMA)

    if 'errors' in validated_input:
        logger.error('Error in input...')
        return {
            'errors': validated_input['errors']
        }

    logger.info('Input validated...')

    valid_input = validated_input['validated_input']

    result = run_inference({"input": valid_input})
    url = upload_video(result)

    return url


if __name__ == "__main__":
    # wait_for_service(url=f'{LOCAL_URL}/health-check')

    print("Cog API Service is ready. Starting RunPod serverless handler...")

    runpod.serverless.start({"handler": handler})
