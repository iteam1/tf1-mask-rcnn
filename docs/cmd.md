- install required packages: 

        pip install labelme
        apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0
        pip install contextlib2
        pip install IPython

- convert the labelme labels training set into COCO format:

        python3 scripts/labelme2coco.py dataset/train \
        --output dataset/train.json

- convert the labelme labels training set into COCO format:

        python3 scripts/labelme2coco.py dataset/val \
        --output dataset/val.json

*Note: Inside docker

- export tfrecord

        python scripts/create_coco_tf_record.py --logtostderr \
        --train_image_dir=dataset/train_img \
        --test_image_dir=dataset/val_img \
        --train_annotations_file=dataset/train.json \
        --test_annotations_file=dataset/val.json \
        --output_dir=dataset

- visualize tfrecord `python3 scripts/visualize_tfrecord.py dataset/train.record dataset/labelmap.pbtxt`

# train model
- export `PYTHONPATH`: `export PYTHONPATH="/mrcnn/models/research"` and `export PYTHONPATH="/mrcnn/models/research/slim"`

        python models/research/object_detection/model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=dataset/mask_rcnn_inception_v2_coco.config

# export model
CUDA_VISIBLE_DEVICES=1 python models/research/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path mask_rcnn_inception_v2_coco.config \
--trained_checkpoint_prefix checkpoints/model.ckpt-163085 --output_directory inference_graph

CUDA_VISIBLE_DEVICES=1 python models/research/object_detection/model_main.py --logtostderr --model_dir=training_crackline/ --pipeline_config_path=crack_line.config

*Note*

        File "/usr/local/lib/python3.6/dist-packages/qtpy/QtGui.py", line 30, in <module>
            from PyQt5.QtGui import *
        ImportError: libGL.so.1: cannot open shared object file: No such file or directory