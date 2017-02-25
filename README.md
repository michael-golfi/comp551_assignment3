# Project Code for Team Highly Classified

## Part One

## Part Two

## Part Three

In order to run the retrain experiment:

```powershell
python tensorflow/tensorflow/examples/image_retraining/retrain.py ^
--bottleneck_dir=bottlenecks ^
--how_many_training_steps 500 ^
--model_dir=model ^
--output_graph=model/retrained_graph.pb ^
--output_labels=model/retrained_labels.txt ^
--image_dir data
```