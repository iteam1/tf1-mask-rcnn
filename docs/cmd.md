- install required packages: 

        pip install labelme && \
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

test exported model: `python scripts/test_tf.py` or `python scripts/test_tf.py`

*Note* 
If you got error: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
Run command: `apt-get install -y libgl1-mesa-dev` and `apt-get install -y libglib2.0-0`

If you got error:

        File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py", line 693, in __init__
            self._session = tf_session.TF_NewSessionRef(self._graph._c_graph, opts)
        tensorflow.python.framework.errors_impl.InvalidArgumentError: Invalid device ordinal value (1). Valid range is [0, 0].
        	while setting up XLA_GPU_JIT device number 1

Add bellow commands to `models/research/object_detection/export_inference_graph.py`

        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" #(or "1" or "2")
