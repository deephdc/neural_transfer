# Image utils

import torch
import subprocess
from os import path
from PIL import Image
import torchvision.transforms as transforms
import neural_transfer.config as cfg

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def download_model(name):
    try:
        nums = [cfg.MODEL_DIR, name]
        model_path = '{0}/{1}.pth'.format(*nums)
        
        if not path.exists(model_path):
            remote_nums = [cfg.REMOTE_MODELS_DIR, name]
            remote_model_path = '{0}/{1}.pth'.format(*remote_nums)
            print('[INFO]: Model not found, downloading model...')
            # from "rshare" remote storage into the container
            command = (['rclone', 'copy', '--progress', remote_model_path, cfg.MODEL_DIR])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            print('[INFO]: Finished.')
        else:
            print("[INFO]: Model found.")
            
    except OSError as e:
        output, error = None, e





