# Project Code for Team Highly Classified

## Preparing the code

First, we need to take the given data files (trainX.npy, trainY.npy, testX.npy) and put them in the base of the project. Then run:

```bash
python ./preprocess.py
```

## Part One

## Part Two

## Part Three

```bash
python retrain.py \
--bottleneck_dir=bottlenecks \
--how_many_training_steps 32000 \
--learning_rate 0.001 \
--model_dir=model \
--output_graph=model/retrained_graph.pb \
--output_labels=model/retrained_labels.txt \
--image_dir data
```