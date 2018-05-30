#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel build tensorflow/tools/graph_transforms:summarize_graph;\
bazel build tensorflow/tools/graph_transforms:transform_graph;\
bazel build tensorflow/contrib/lite/toco:toco \
bazel build tensorflow/examples/label_image:label_image;${KTO}"
