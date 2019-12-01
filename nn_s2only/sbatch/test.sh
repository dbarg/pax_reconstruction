#!/bin/bash

python ./nn_s2waveforms_xy_train.py -directory ../../xe1t-processing/pax_merge/temp_s2/ -max_dirs 11 -events_per_batch 100 -downsample 10
