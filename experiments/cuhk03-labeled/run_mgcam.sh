#!/usr/bin/env sh
LOG=./mgcam-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=/path-to-caffe/build/tools/caffe

$CAFFE train --solver=./solver_mgcam.prototxt --gpu=0 2>&1 | tee $LOG
