import tensorflow as tf
import numpy as np
import os
import sys
import subprocess

from rod.helper import get_model_list

ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR,'thesis_tests/NMS_TESTING')
TF_DIR = '/home/gustav/workspace/tensorflow/tf_models/research/object_detection'

MODELS = get_model_list(MODELS_DIR)

for model in MODELS:
    model_path = MODELS_DIR + "/" + model
    cmd = 'python {}/eval.py \
            --logtostderr \
            --pipeline_config_path={}/pipeline.test.config \
            --checkpoint_dir={}/ \
            --eval_dir={}/eval'.format(TF_DIR,model_path,model_path,model_path)

    print ("> Evaluating model {}".format(model))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print process.returncode

print("> Evaluation complete")
