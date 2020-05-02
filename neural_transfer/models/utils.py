# Image utils

import torch
import subprocess
import os
import sys
from six.moves import urllib
from os import path
from os import listdir
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

def download_pred_model(name):
    nums = [cfg.neural_RemoteShare, name]
    remote_model_path ='{0}{1}.pth'.format(*nums)

    nums = [cfg.MODEL_DIR, name]
    file_path = '{0}/{1}.pth'.format(*nums)
    
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s model %.1f' % (name,
                        float(count * block_size)))
        sys.stdout.flush()

    file_path, _ = urllib.request.urlretrieve(remote_model_path, file_path, _progress)
    statinfo = os.stat(file_path)
    
    if os.path.exists(file_path):
        print('[INFO] Successfully downloaded %s model, %d bytes' % 
               (name, statinfo.st_size))
        dest_exist = True
        error_out = None
    else:
        dest_exist = False
        error_out = '[ERROR, url_download()] Failed to download ' + name + \
                    ' model from ' + url_path
        
    return dest_exist, error_out


def download_model(name):
    try:
        nums = [cfg.MODEL_DIR, name]
        model_path = '{0}/{1}.pth'.format(*nums)
        
        if not path.exists(model_path):
            remote_nums = [cfg.REMOTE_MODELS_DIR, name]
            remote_model_path = '{0}/{1}.pth'.format(*remote_nums)
            print('[INFO] Model not found, trying to download model...')
            # from "rshare" remote storage into the container
            command = (['rclone', 'copy', '--progress', remote_model_path, cfg.MODEL_DIR])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
        else:
            print("[INFO] Model found.")
            
    except OSError as e:
        output, error = None, e

#Loads Dataset from  Nextcloud.
def download_dataset():
    try:
        images_path = os.path.join(cfg.DATA_DIR, "raw/training_dataset") 
        
        if not path.exists(images_path):
            print('[INFO] No data found, trying to download training dataset...')
            # from "rshare" remote storage into the container
            command = (['rclone', 'copy', '--progress', cfg.REMOTE_IMG_DATA_DIR, images_path])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
        else:
            print("[INFO] Training dataset folder already exists.")
            
    except OSError as e:
        output, error = None, e
        

def download_style_image(name):
    try:
        images_path = cfg.DATA_DIR
        
        nums = [cfg.REMOTE_IMG_STYLE_DIR, name]
        model_path = '{0}/{1}'.format(*nums)
        
        print('[INFO] Downloading image...')
        # from "rshare" remote storage into the container
        command = (['rclone', 'copy', '--progress', cfg.REMOTE_IMG_STYLE_DIR, images_path])
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
        print('[INFO] Finished.')
            
    except OSError as e:
        output, error = None, e
        
        
def upload_model(model_path):
    try:      
        #from the container to "rshare" remote storage 
        command = (['rclone', 'copy', '--progress', model_path, cfg.REMOTE_MODELS_DIR])
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
    except OSError as e:
        output, error = None, e

def get_models():
    models = []
    for f in listdir(cfg.MODEL_DIR): 
        if f.endswith(".pth") and str(f) not in ['mosaic.pth', 'udnie.pth', 'candy.pth', 'rain_princess.pth']:
            models.append(f[:-4])
    return models
