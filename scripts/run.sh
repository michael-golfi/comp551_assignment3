#!/bin/bash

python retrain.py \
--bottleneck_dir=bottlenecks \
--how_many_training_steps 32000 \
--learning_rate 0.001 \
--model_dir=model \
--output_graph=model/retrained_graph.pb \
--output_labels=model/retrained_labels.txt \
--image_dir data
