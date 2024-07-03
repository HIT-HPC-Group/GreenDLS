from cuda import cudart
import cv2
import os
import tensorrt as trt
import torch 
from time import time
import numpy as np
from torchvision import transforms, datasets
from glob import glob
import json
import pynvml
import sys
sys.path.append(r"../data")
from imagenet import *
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
nHeight=224
nWidth=224
batch_size=256
MaxBatchSize=256
trtFile = "../model/vgg19.plan"
onnxFile= "../model/vgg19.onnx"
data_transform = {
    "test": transforms.Compose([transforms.Resize([224,224]),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


test_dataset = TinyImageNet("../data/imagenet/tiny-imagenet-200", train=False,transform=data_transform["test"])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
train_num = len(test_dataset)
print(train_num)





# #构建logger,builder,network
logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#构建parser
parser = trt.OnnxParser(network, logger)
# config = builder.create_builder_config()

#读入onnx查看有无错误
if not os.path.exists(onnxFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
profile0 = builder.create_optimization_profile()
# profile1 = builder.create_optimization_profile()
# profile2 = builder.create_optimization_profile()
# profile3 = builder.create_optimization_profile()
# 最小和最大，指运行时，允许的最小和最大的范围
# 最佳值，用于选择内核，这里通常为运行时，最期望的大小
profile0.set_shape(inputTensor.name, (1, 3, nHeight, nWidth), (16, 3, nHeight, nWidth), (MaxBatchSize, 3, nHeight, nWidth))
# profile1.set_shape(inputTensor.name, (1, 3, nHeight, nWidth), (16, 3, nHeight, nWidth), (MaxBatchSize, 3, nHeight, nWidth))
# profile2.set_shape(inputTensor.name, (1, 3, nHeight, nWidth), (16, 3, nHeight, nWidth), (MaxBatchSize, 3, nHeight, nWidth))
# profile3.set_shape(inputTensor.name, (1, 3, nHeight, nWidth), (16, 3, nHeight, nWidth), (MaxBatchSize, 3, nHeight, nWidth))
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) 
config.add_optimization_profile(profile0)
# config.add_optimization_profile(profile1)
# config.add_optimization_profile(profile2)
# config.add_optimization_profile(profile3)
#生成并序列化engine
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)

    
with open(trtFile, "rb") as f:
    engineString=f.read()
#反序列化构建tensorrt引擎
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engineString)
#构建执行上下文环境
context = engine.create_execution_context()
#这句非常重要！！！定义batch为动态维度,设置输入维度大小
context.set_binding_shape(0, [batch_size, 3, nHeight, nWidth])
  
sum_time=0
acc = 0
n = 0
output=[]

_, stream = cudart.cudaStreamCreate() 
for xTest, yTest in test_loader:
    # print(xTest.numpy().shape)
    inputH0 = np.ascontiguousarray(xTest.numpy().reshape(-1))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    # 完整一次推理
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeStart = time()
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    outputH0=torch.max(torch.from_numpy(outputH0), dim=1)[1]
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    t=(trtTimeEnd - trtTimeStart) * 1000
    sum_time+=t
    if(len(outputH0)>len(yTest)):
        outputH0=outputH0[0:len(yTest)]
    acc += torch.sum(outputH0 ==yTest).sum().item()
    n += xTest.shape[0]
    print("time: %fms acc: %f "%(t ,acc/n))
print("avgtime: %fms acc: %f "%(batch_size*sum_time/n ,acc/n))


print("Succeeded running model in TensorRT!")