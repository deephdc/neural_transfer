# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields, validate
from marshmallow import Schema, INCLUDE

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data/')
IMG_STYLE_DIR = os.path.join(IN_OUT_BASE_DIR, 'neural_transfer/dataset/style_images')
MODEL_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')

neural_RemoteSpace = 'rshare:/neural_transfer/'
neural_RemoteShare = 'https://nc.deep-hybrid-datacloud.eu/s/9Qp4mxNBaLKmqAQ/download?path=%2F&files='

REMOTE_IMG_DATA_DIR = os.path.join(neural_RemoteSpace, 'dataset/training_dataset/')
REMOTE_IMG_STYLE_DIR = os.path.join(neural_RemoteSpace, 'styles/')

REMOTE_MODELS_DIR = os.path.join(neural_RemoteSpace, 'models/')

# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    
    img_content = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="image_content",
        location="form",
        description="Image to be styled."
    )
            
    accept = fields.Str(
            require=False,
            description="Returns the image with the new style or a pdf containing the 3 images.",
            missing='image/png',
            validate=validate.OneOf(['image/png', 'application/pdf']))
    
    model_name = fields.Str(
        required=False,
        missing = "mosaic",
        description="Name of the saved model. This module already comes with some styles, just write the name: 'mosaic', 'candy', 'rain_princess' or 'udnie'. You can see the styles in the dataset/style_images folder. Running 'get_metadata' return the list of models in the module."
    )
    
# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter
        
    model_name = fields.Str(
        required=True,
        description="Name of the style image e.g. 'name.jpg' in nextcloud. This will also be the name of the model."
    )
    
    upload_model = fields.Boolean(
        required=False,
        missing = 2,
        description="Upload model to nextcloud."
    )
    
    epochs = fields.Int(
        required=False,
        missing = 2,
        description="Number of training epochs."
    )
    
    learning_rate = fields.Float(
        required=False,
        missing = 0.003,
        description="Learning rate."
    )
    
    batch_size = fields.Int(
        required=False,
        missing = 4,
        description="Batch size for training."
    )
    
    content_weight = fields.Float(
        required=False,
        missing = 1e5,
        description="Weight for content-loss."
    )
    
    style_weight = fields.Float(
        required=False,
        missing = 1e10,
        description="Number of iterations on the network to compute the gradients."
    )
    
    size_train_img = fields.Int(
        required=False,
        missing = 256,
        description="Size of training images, default is 256 X 256"
    )
    
    log_interval = fields.Int(
        required=False,
        missing = 200,
        description="Number of images after which the training loss is logged."
    )
    
    


