# Highly Classified Project Code

## Preparing the code

```bash
wget https://inclass.kaggle.com/c/comp-551-tiny-imagenet/download/tinyX.npy
wget https://inclass.kaggle.com/c/comp-551-tiny-imagenet/download/tinyX_test.npy
wget https://inclass.kaggle.com/c/comp-551-tiny-imagenet/download/tinyY.npy
python ./preprocess.py
```

## Part One

python logistic_regression.py

## Part Two

python part2.py

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