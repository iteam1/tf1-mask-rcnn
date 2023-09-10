- install required packages: 

        pip install labelme && \
        apt-get update && \
        apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0 && \
        pip install contextlib2 && \
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

- export `PYTHONPATH`: `export PYTHONPATH="/mrcnn/models/research" && export PYTHONPATH="/mrcnn/models/research/slim"`

- train model: `python models/research/object_detection/model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=dataset/mask_rcnn_inception_v2_coco.config`

- export model

        python models/research/object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path mask_rcnn_inception_v2_coco.config \
        --trained_checkpoint_prefix training/model.ckpt-xxx --output_directory inference_graph