# setup docker
- pull `tensorflow v1` docker image: `docker pull <YOUR_DOCKER_IMAGE_NAME>` example `docker pull tensorflow/tensorflow:1.15.5-gpu-py3`

                tensorflow/tensorflow   1.15.5-gpu-py3   73be11373498   2 years ago    3.61GB
                tensorflow/tensorflow   1.13.2-gpu-py3   b5408c35298a   4 years ago    3.35GB
                tensorflow/tensorflow   1.14.0-gpu-py3   a7a1861d2150   4 years ago    3.51GB

- build docker container name `mytf1`, mount your working directory to folder `mrcnn` inside container: `docker run --gpus all --shm-size=30g --name mytf1 -it -v $PWD:/mrcnn <YOUR_DOCKER_IMAGE_NAME> /bin/bash`, example: `docker run --gpus all --shm-size=30g --name mytf1 -it -v $PWD:/mrcnn tensorflow/tensorflow:1.15.5-gpu-py3 /bin/bash`

- start docker container by command: `docker start -it mytf1`

- stop docker container by command: `docker stop mytf1`

- remove docker container by command: `docker rm mytf1`

- remove docker image by command `docker rmi <YOUR_DOCKER_IMAGE_NAME>`

# Install TensorFlow Object Detection API.
- clone source code `tensorflow object-detection` on branch `r1.13.0` (*You can do this command outside the container*): `git clone -b r1.13.0 --single-branch https://github.com/tensorflow/models.git`

- go inside docker container by command: `docker exec -it mytf1 /bin/bash`

- check gpu and container connection `nvidia-smi`

- install essential packages `pip install --upgrade pip setuptools wheel`

- install protobuf compiler: `apt install protobuf-compiler`

- go to directory `research` directory: `cd models/research`

- compile protoc file to python file: `protoc object_detection/protos/*.proto --python_out=.`

- install `opencv-python`: `pip install opencv-python==4.1.2.30`

- install `object-detection` by `setup.py` file: `python -m pip install .`

- export `PYTHONPATH`: `export PYTHONPATH="/mrcnn/models/research" && export PYTHONPATH="/mrcnn/models/research/slim"`

- test your installation: `python object_detection/builders/model_builder_test.py`

# Install pycocoapi
- clone coco: `git clone https://github.com/pdollar/coco.git`

- build `pycocotools`:

        cd coco/PythonAPI && \
        make && \
        make install && \
        python setup.py install

*Note:*

Error:

        File "/usr/local/lib/python3.6/dist-packages/object_detection/models/keras_models/resnet_v1.py", line 21, in <module>
            from keras.applications import resnet
        ImportError: cannot import name 'resnet'

change line 21 `/usr/local/lib/python3.6/dist-packages/object_detection/models/keras_models/resnet_v1.py` from `from keras.applications import resnet` to `from keras.applications import resnet50 as resnet`

Keras versions:

        from versions: 0.2.0, 0.3.0, 0.3.1, 0.3.2, 0.3.3, 1.0.0, 1.0.1, 1.0.2, 1.0.3, 1.0.4, 1.0.5, 1.0.6, 1.0.7, 1.0.8, 1.1.0, 1.1.1, 1.1.2, 1.2.0, 1.2.1, 1.2.2, 2.0.0, 2.0.1, 2.0.2, 2.0.3, 2.0.4, 2.0.5, 2.0.6, 2.0.7, 2.0.8, 2.0.9, 2.1.0, 2.1.1, 2.1.2, 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.2.0, 2.2.1, 2.2.2, 2.2.3, 2.2.4, 2.2.5, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.4.2, 2.4.3, 2.5.0rc0, 2.6.0rc0, 2.6.0rc1, 2.6.0rc2, 2.6.0rc3, 2.6.0, 2.7.0rc0, 2.7.0rc2, 2.7.0, 2.8.0rc0, 2.8.0rc1, 2.8.0, 2.9.0rc0, 2.9.0rc1, 2.9.0rc2, 2.9.0, 2.10.0rc0, 2.10.0rc1, 2.10.0