#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
 --in_graph=${IN_GRAPH} \
 --out_graph=${OUT_GRAPH} \
 --inputs=${INPUTS} \
 --outputs=${OUTPUTS} \
 --transforms=${TRANSFORMS[@]} \
 2>&1 | tee ${RESULTS_PATH}/transform_${MODEL_NAME}.txt;${KTO}"
