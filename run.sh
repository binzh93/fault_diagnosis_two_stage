#!/bin/sh

# python train_and_test.py --data_dir "fault_data"


cd /workspace/mnt/group/face/zhubin/alg_code/fault_diagnosis/fault_diagnosis_two_stage
pip2 install tensorflow-gpu --index-url https://pypi.douban.com/simple

python2 train_and_test.py