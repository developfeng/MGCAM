#!/usr/bin/env sh
LOG=./mgcam-siamese`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=/path-to-caffe/build/tools/caffe

$CAFFE train --solver=./solver_mgcam_siamese.prototxt --weights=./mgcam_iter_75000.caffemodel --gpu=0 2>&1 | tee $LOG
