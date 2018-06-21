#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${ROOT_PATH}/test_images/000000581781.jpg \
--input_layer=${INPUTS} \
--output_layer=${OUTPUTS} \
--graph=${IN_GRAPH} \
--labels=${ROOT_PATH}/rod/data/mscoco_label_map.pbtxt \
2>&1 | tee ${RESULTS_PATH}/min_max_log_${MODEL_NAME}.txt;${KTO}"
