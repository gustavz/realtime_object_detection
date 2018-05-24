#!/bin/bash

# written and copyright by
# www.github.com/GustavZ

gnome-terminal -x sh -c "cd ${TF_PATH};\
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
 --in_graph=${IN_GRAPH} \
 --out_graph=${OUT_GRAPH} \
 --inputs=${INPUTS} \
 --outputs=${OUTPUTS} \
 --transforms=${TRANSFORMS};bash"

cd ${TF_PATH}
 bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=${IN_GRAPH} \
  --out_graph=${OUT_GRAPH} \
  --inputs=${INPUTS} \
  --outputs=${OUTPUTS} \
  --transforms='
  strip_unused_nodes(type=float, shape="1,400,400,3")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants
  fold_batch_norms
  quantize_weights
  strip_unused_nodes
  sort_by_execution_order
  fuse_resize_pad_and_conv
  '
