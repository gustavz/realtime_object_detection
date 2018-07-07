import tensorflow as tf
import numpy as np
import os
import sys
import subprocess

from rod.helper import get_model_list

ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR,'test')
TF_DIR = '/home/gustav/workspace/tensorflow/tf_models/research/object_detection

MODELS = get_model_list(MODELS_DIR)
MODEL_VERSIONS = ['300p','200p','100p','50p','10p']

for model in MODELS:
    model_path = MODELS_DIR + "/" + model
    for version in MODEL_VERSIONS:
        cmd = 'python {}/export_inference_graph.py \
                --input_type=image_tensor \
                --pipeline_config_path={}/pipeline.{}.config \
                --trained_checkpoint_prefix={}/model.ckpt \
                --output_directory={}'.format(TF_DIR,
                                            model_path,version,
                                            model_path,
                                            model_path+"_"+version)

        print ("> Exporting model {}".format(model))
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print process.returncode
