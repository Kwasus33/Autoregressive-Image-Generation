import os

import requests
from requests import HTTPError

# repo with pretrained PixelCNN impl
BASE_URL = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial12/"
PRETRAINED_FILES = ["PixelCNN.ckpt"]
# model_path = "../../data"


def load_pixel_cnn(base_url: str, model_path: str, pretrained_files: list[str]):
    os.makedirs(model_path, exist_ok=True)

    for file_name in pretrained_files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                with requests.get(file_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                fh.write(chunk)
            except HTTPError as e:
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                    e,
                )
