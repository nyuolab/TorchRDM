import re
import shutil
from pathlib import Path

import scipy.io as sio


def process_images(source_path: Path = Path("./"), store_path: Path = Path("./")):
    """Preprocess the image data for later use.

    Parameters
    ----------
    source_path : Path
        The source path of data.
    store_path : Path
        Where to store the preprocessed data.
    """
    # First create a directory for mooney images and grayscale images
    img_path = store_path / "imgs/"
    img_path.mkdir(parents=True, exist_ok=True)

    # Read the rdm label
    loaded = sio.loadmat(source_path / "RDMs" / "RDM_label_order.mat")["labelnames"]
    loaded = [re.split(r"'|_", str(item[0]))[-2] for item in loaded]

    # Then copy the files and rename, according to the rdm labels
    for file in source_path.glob("**/*.bmp"):

        name = file.name
        image_raw_name_parsed = re.split(r"-|\.", name)
        if len(image_raw_name_parsed) < 3:
            continue
        image_raw_name = image_raw_name_parsed[2]

        if image_raw_name in loaded:
            index = loaded.index(image_raw_name)
            if name.startswith("filter"):
                prefix = ""
            elif name.startswith("test"):
                prefix = "-"
            else:
                continue
            shutil.copy(file, img_path / f"{prefix}{index}.bmp")
