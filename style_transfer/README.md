# Style Transfer - Tensorflow 2.0
![alt text](https://github.com/jastern33/general/raw/master/style_transfer/demos/kandinsky-turtle.png)

This is a project demoing the Tensorflow 2.0 style transfer tutorial. My purpose in this project was primarily to familiarize myself with TF 2.0, so most of the code is from TensorFlow's official tutorial [here](https://www.tensorflow.org/beta/tutorials/generative/style_transfer). I've added comments and docstrings and rewritten their Jupyter Notebook as a python program that can be run from the command line.
# Usage
The easiest way to run this code is in a docker container. The only requirements are [Docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Here are the necessary docker commands:

Build the docker image (takes awhile): 

`docker build -t tf-2.0 -f Dockerfile.tf . --rm`

Run a docker container based on that image: 

`nvidia-docker run -it --name style-transfer-tf --rm -v /my/machine/image/folder:/images tf-2.0`

Then run the following command in the docker container for a basic demo: 

`python3 style_transfer_main`

To see additional options run: 

`python3 style_transfer_main --help`
