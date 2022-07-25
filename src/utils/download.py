import zipfile
import requests

import shutil

from typing import Union
from pathlib import Path

import tempfile

RDM_URL = "https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvNDE4NjEvZWxpZmUtNDE4NjEtY29kZTEtdjIuemlw/elife-41861-code1-v2.zip?_hash=3NAXgAwxMozII5Z% 2F3qkXJ1f%2BuDHnlI5XeNQGIgBLBZU%3D"
IMAGES_URL = ""

def download_rdm(store_path: Path):
    # Operate within the Temporary Directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # First download the RDM data
        r = requests.get(RDM_URL)
        with open(tmp_dir / "rdm.zip", "wb") as f:
            f.write(r.content)
        
        # Then unzip it
        with zipfile.ZipFile(tmp_dir / "rdm.zip", "r") as zip_ref:
            # Create the directory
            zip_ref.extractall(tmp_dir)

        # Copy the interested files to store path
        shutil.copy(
            tmp_dir / "eLife_Final_Reduced" / "Data" / "Processed" / "Figure6" / "FMRI_Model_Fusion.mat",
            store_path / "FMRI_Model_Fusion.mat"
        )
        shutil.copy(
            tmp_dir / "eLife_Final_Reduced" / "Data" / "Processed" / "Figure6" / "MEG_Model_Fusion.mat",
            store_path / "MEG_Model_Fusion.mat"
        )

def download_images(store_path: Path):
    # Operate within the Temporary Directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # First download the RDM data
        r = requests.get(RDM_URL)
        with open(tmp_dir / "rdm.zip", "wb") as f:
            f.write(r.content)
        
        # Then unzip it
        with zipfile.ZipFile(tmp_dir / "rdm.zip", "r") as zip_ref:
            # Create the directory
            zip_ref.extractall(tmp_dir)

        # Copy the interested files to store path
        shutil.copy(
            tmp_dir / "eLife_Final_Reduced" / "Data" / "Processed" / "Figure6" / "FMRI_Model_Fusion.mat",
            store_path / "FMRI_Model_Fusion.mat"
        )
        shutil.copy(
            tmp_dir / "eLife_Final_Reduced" / "Data" / "Processed" / "Figure6" / "MEG_Model_Fusion.mat",
            store_path / "MEG_Model_Fusion.mat"
        )


