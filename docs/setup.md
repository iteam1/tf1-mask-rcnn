# setup docker
- pull docker: `docker pull tensorflow/tensorflow:1.14.0-gpu-py3`

- pull docker: `docker pull tensorflow/tensorflow:1.15.5-gpu-py3`

- build docker container name `mytf1`, mount your working directory to folder `mrcnn` inside container: `docker run --gpus all --shm-size=30g --name mytf1 -it -v $PWD:/mrcnn <YOUR_DOCKER_IMAGE_NAME> /bin/bash`, example: `docker run --gpus all --shm-size=30g --name mytf1 -it -v $PWD:/mrcnn tensorflow/tensorflow:1.15.5-gpu-py3 /bin/bash`

- start docker container by command: `docker start -it mytf1`

- stop docker container by command: `docker stop mytf1`

- remove docker container by command: `docker rm mytf1`

- remove docker image by command `docker rmi <YOUR_DOCKER_IMAGE_NAME>`

# Install TensorFlow Object Detection API.
- go inside docker container by command: `docker exec -it mytf1 /bin/bash`

- check gpu and container connection `nvidia-smi`

- install essential packages `pip install --upgrade pip setuptools wheel`

- clone source code `tensorflow object-detection`: `git clone https://github.com/tensorflow/models.git`

docker exec -it <your_container> bin/bash

apt install protobuf-compiler

cd models/research

protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf1/setup.py .

pip3 install opencv-python==4.1.2.30

python -m pip install .

pip install keras==2.1.6

*Note:*
        File "/usr/local/lib/python3.6/dist-packages/object_detection/models/keras_models/resnet_v1.py", line 21, in <module>
            from keras.applications import resnet
        ImportError: cannot import name 'resnet'

# download pretrained model tensorflow1
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

# Test the installation (custom)
python object_detection/builders/model_builder_tf1_test.py