#!/usr/bin/env bash
echo "Starting in`pwd`"

# venv
cd ..
source venv/bin/activate
echo "Now in`pwd`"

# venv
cd tensorflow-for-poets-2
echo "And now in`pwd`"


export IMAGE_SIZE=224
export ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

# Run tensorboard in background, kill it if already launched
pkill -f "tensorboard"
tensorboard --logdir tf_files/training_summaries &

# Training
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/puzzle_pieces
