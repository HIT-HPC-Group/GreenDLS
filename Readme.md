## GREENDLS

`./data`  holds the inference data needed for deep learning applications

`./llm` is the source code of our work in Qwen

`./src` is the source code of our work.

`./PerformanceMeasurement` is our tool to measure GPU power, GPU energy, GPU SM frequency, GPU memory frequency, execution time, CPU power, CPU energy, etc.

`./profile` is used to collect data and build predictive models

## Installation and Deployment Process

1. ### Installation of Libraries and Software: 

   - Install docker.

   ```
   apt install docker.io
   ```

   - Install NVIDIA Container Toolkit.

   ```
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
      && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
   ```

   - Pull docker images.

   ```
   docker pull txylabs/qwen
   docker pull txylabs/tensorrt
   ```

2. ### Deployment of Code:

   - Pull GREENDLS code.

   ```
   git clone https://github.com/HIT-HPC-Group/GreenDLS.
   ```

   - Compile measurement tool

     - If you are deploying a normal DNN model, such as resnet50,vgg19,densenet201

     ```
     sudo docker run -it --privileged=true --shm-size="32g"  -v /yoursource:/home --gpus all txylabs/tensorrt
     cd /home/PerformanceMeasurement
     make
     ```
   
     - If you want to deploy Qwen language model  
   
     ```
     sudo docker run -it --privileged=true --shm-size="32g"  -v /yoursource:/home --gpus all txylabs/qwen
     cd /home/PerformanceMeasurement
     make
     ```
   
   - Convert model.
     - If you are deploying a normal DNN model, such as resnet50,vgg19,densenet201, just train the corresponding pytorch model and save it in onnx  format, then convert it to a trt engine model using model/onnxtoolan.py.
   
     - If you want to deploy Qwen language model, refer to https://github.com/QwenLM/Qwen.git. We used the Qwen-7B-Chat-Int4 version in our paper  	
   
   -  Prepare the data set.  
     - For resnet50 vgg19 densenet201 model, from https://www.kaggle.com/c/tiny-imagenet/data to download data set to the ./data
     - For Qwen ,download data set from [IWSLT英中机器翻译口语测试集 · 数据集 (modelscope.cn)](https://www.modelscope.cn/datasets/iic/IWSLT-English-to-Chinese-Machine-Translation-Spoken/)[IWSLT英中机器翻译口语测试集 · 数据集 (modelscope.cn)](https://www.modelscope.cn/datasets/iic/IWSLT-English-to-Chinese-Machine-Translation-Spoken/) to ./data
   
   
   ### 3.Getting Started:
   
   - Start Docker
   
     - If you are deploying a normal DNN model, such as resnet50,vgg19,densenet201
   
       ```
       sudo docker run -it --privileged=true --shm-size="32g"  -v /yoursource:/home --gpus all txylabs/tensorrt
       ```
   
     - If you want to deploy Qwen language model  
   
       ```
       sudo docker run -it --privileged=true --shm-size="32g"  -v /yoursource:/home --gpus all txylabs/qwen
       ```
   
   - Measure
   
     Use the ./profile/trtInference.py file to collect the running status of DNN on the GPU.
   
   - Profiler
   
     Use the ./profile/predictModel.ipynb file to build a prediction model
   
   - train && test
   
     Run ./src/main.sh for DNN model and ./llm/src/main.sh for Qwen.

​			 