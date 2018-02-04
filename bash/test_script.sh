#!/usr/bin/env bash
echo "Starting in`pwd`"

# venv
cd ..
source venv/bin/activate
echo "Now in`pwd`"

cd tensorflow-for-poets-2
echo "And now in`pwd`"


python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=../images/picasso.jpg #tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg