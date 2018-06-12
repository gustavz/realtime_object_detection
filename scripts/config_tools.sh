#!/bin/bash
#
# written and copyright by
# www.github.com/GustavZ
#
# for additional information see:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs


### MODEL AND SYSTEM CONFIG ###
### CHANGE THIS ACCORDING TO YOUR SYSTEM ###
export MODEL_NAME="mask_rcnn_mobilenet_v1_224_coco_person"
export TF_PATH="${HOME}/workspace/tensorflow/tensorflow"
export ROOT_PATH="${HOME}/workspace/realtime_object_detection"
export USE_OPTIMIZED=false
export KEEP_TERMINALS_OPEN=false

##########################
### DO NOT CHANGE THIS ###
##########################
export MODEL_PATH="${ROOT_PATH}/models/${MODEL_NAME}"
export IN_GRAPH="${MODEL_PATH}/frozen_inference_graph.pb"
export OUT_GRAPH="${MODEL_PATH}/optimized_inference_graph.pb"
export TFLITE_GRAPH="${MODEL_PATH}/frozen_inference_graph.tflite"
export RESULTS_PATH="${ROOT_PATH}/test_results"
# conditionals
if [ ${USE_OPTIMIZED} = true ] ; then
export MODEL_NAME="${MODEL_NAME}_optimized"
export IN_GRAPH=${OUT_GRAPH}
fi
if [ ${KEEP_TERMINALS_OPEN} = true ] ; then
export KTO=bash
else
export KTO=""
fi

### MODEL TRANSFORMATION CONFIG ###
### CHANGE THIS ACCORDING TO YOUR MODEL ###
#export SHAPE='\"1,513,384,3\"' # DeepLab
export TSHAPE='\"1,224,224,3\"' # used in transform_graph.sh
export SHAPE="1,224,224,3" # used in benchmark_model.sh
export STD_VALUE=127.5
export MEAN_VALUE=127.5
export INPUT_TYPE='uint8'
export LOGGING='\"__requant_min_max:\"'
#export INPUTS='ImageTensor' #DeepLab
export INPUTS='image_tensor' #Object_detection
#export OUTPUTS='SemanticPredictions' #DeepLab
export OUTPUTS='num_detections,detection_boxes,detection_scores,detection_classes,detection_masks' #Object_detection
export TRANSFORMS=("'
add_default_attributes
remove_device
strip_unused_nodes(type=float, shape='${TSHAPE}')
remove_nodes(op=Identity, op=CheckNumerics, op=BatchNorm)
fold_constants
fold_batch_norms
sort_by_execution_order
'")
#freeze_requantization_ranges(min_max_log_file=${RESULTS_PATH}/min_max_log_${MODEL_NAME}.txt)
#insert_logging(op=RequantizationRange, show_name=true, message='${LOGGING}')

########################
echo "> doublecheck set variables:"
echo "MODEL_NAME: $MODEL_NAME"
echo "TF_PATH: $TF_PATH"
echo "ROOT_PATH: $ROOT_PATH"
echo "IN_GRAPH: ${IN_GRAPH}"
echo "Use Optimized Model: ${USE_OPTIMIZED}"
echo "Keep Terminals Open: ${KEEP_TERMINALS_OPEN}"

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
