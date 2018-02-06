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
    --image=tf_files/test_set/merge_test/merge1.jpg