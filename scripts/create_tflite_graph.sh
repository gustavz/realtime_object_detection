#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=${IN_GRAPH} \
  --output_file=${TFLITE_GRAPH} \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=${SHAPE} \
  --input_array=${INPUTS} \
  --output_array=${OUTPUTS} \
  --std_value=${STD_VALUE} --mean_value=${MEAN_VALUE} \
  2>&1 | tee ${RESULTS_PATH}/create_tflite_${MODEL_NAME}.txt;${KTO}"
