# convert labelme3.0 to coco
python3 scripts/labelme3_labelme.py dataset/train
python3 scripts/labelme3_labelme.py dataset/test

*Note: Outside the docker

python3 scripts/labelme2coco.py dataset/train \
--output dataset/train.json

python3 scripts/labelme2coco.py dataset/test \
--output dataset/test.json

*Note: Inside docker

# export tfrecord
python scripts/create_coco_tf_record.py --logtostderr \
--train_image_dir=dataset/train_img \
--test_image_dir=dataset/test_img \
--train_annotations_file=dataset/train.json \
--test_annotations_file=dataset/test.json \
--output_dir=dataset

# visualize tfrecord
python3 scripts/visualize_tfrecord.py dataset/train.record dataset/labelmap.pbtxt

# train model
CUDA_VISIBLE_DEVICES=1 python models/research/object_detection/model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=mask_rcnn_inception_v2_coco.config

# export model
CUDA_VISIBLE_DEVICES=1 python models/research/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path mask_rcnn_inception_v2_coco.config \
--trained_checkpoint_prefix checkpoints/model.ckpt-163085 --output_directory inference_graph

CUDA_VISIBLE_DEVICES=1 python models/research/object_detection/model_main.py --logtostderr --model_dir=training_crackline/ --pipeline_config_path=crack_line.config
