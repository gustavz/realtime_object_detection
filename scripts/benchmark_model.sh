#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel run tensorflow/tools/benchmark:benchmark_model -- \
 --graph=${IN_GRAPH} \
 --input_layer=${INPUTS} \
 --input_layer_type=${INPUT_TYPE} \
 --input_layer_shape=${SHAPE} \
 --output_layer=${OUTPUTS} \
 --show_run_order=true \
 --show_time=true \
 --show_memory=true \
 --show_summary=true \
 --show_flops=true \
 2>&1 | tee ${RESULTS_PATH}/benchmark_${MODEL_NAME}.txt;${KTO}"
