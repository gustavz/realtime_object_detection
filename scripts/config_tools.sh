#!/bin/bash
#
# written and copyright by
# www.github.com/GustavZ
#
# for additional information see:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs


### MODEL AND SYSTEM CONFIG ###
### CHANGE THIS ACCORDING TO YOUR SYSTEM ###
export MODEL_NAME="mask_rcnn_mobilenet_v1_400_coco_117k"
export TF_PATH="/home/gustav/workspace/tensorflow/tensorflow"
export REPO_PATH="/home/gustav/workspace/realtime_object_detection"
### DO NOT CHANGE THIS ###
export MODEL_PATH="${REPO_PATH}/models/${MODEL_NAME}"
export IN_GRAPH="${MODEL_PATH}/frozen_inference_graph.pb"
export OUT_GRAPH="${MODEL_PATH}/optimized_inference_graph.pb"
export TFLITE_GRAPH="${MODEL_PATH}/frozen_inference_graph.tflite"

### MODEL TRANSFORMATION CONFIG ###
### CHANGE THIS ACCORDING TO YOUR MODEL ###
export SHAPE='1,400,400,3'
export STD_VALUE=127.5
export MEAN_VALUE=127.5
export INPUT_TYPE='uint8'
export INPUTS='image_tensor'
export OUTPUTS='num_detections,detection_boxes,detection_scores,detection_classes,detection_masks'
export TRANSFORMS='
add_default_attributes
strip_unused_nodes(type=float, shape=\"1,400,400,3\")
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
quantize_weights
quantize_nodes
strip_unused_nodes
sort_by_execution_order
'

########################
echo "> doublecheck all paths:"
echo "MODEL_NAME: $MODEL_NAME"
echo "TF_PATH: $TF_PATH"
echo "REPO_PATH: $REPO_PATH"

########################
# Possible Transforms are:
: <<'COMMENT'
add_default_attributes
backport_concatv2
backport_tensor_array_v3
flatten_atrous_conv
fold_batch_norms
fold_constants
fold_old_batch_norms
freeze_requantization_ranges
fuse_pad_and_conv
fuse_remote_graph
fuse_resize_and_conv
fuse_resize_pad_and_conv
insert_logging
merge_duplicate_nodes
obfuscate_names
place_remote_graph_arguments
quantize_nodes
quantize_weights
remove_attribute
remove_control_dependencies
remove_device
remove_nodes
rename_attribute
rename_op
rewrite_quantized_stripped_model_for_hexagon
round_weights
set_device
sort_by_execution_order
sparsify_gather
strip_unused_nodes
COMMENT
