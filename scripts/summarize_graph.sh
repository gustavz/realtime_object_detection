#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
 --in_graph=${IN_GRAPH};bash"
