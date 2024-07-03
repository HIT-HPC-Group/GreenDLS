from cuda import cudart
import cv2
import os
import tensorrt as trt
import torch 
import time
import numpy as np
from torchvision import transforms, datasets
from glob import glob
import json
import pynvml
import sys
import logging
sys.path.append(r"/home/data")
from imagenet import *
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
nHeight=224
nWidth=224
# batchsizelist=[x for x in range(4,257,4)]
# batchsizelist.append(1)
# batchsizelist.append(2)
# batchsizelist.append(4)
# batchsizelist.append(8)
# batchsizelist.sort()
batchsizelist=[256,128,64,32,16,8,4,2,1]
# frequencylist=[900, 840, 780, 720, 660, 600, 540]
frequencylist=[2100, 2040, 1980, 1920, 1860, 1800, 1740, 1680, 1620, 1560, 1500, 1440, 1380, 1320, 1260, 1200, 1140, 1080, 1020, 960,900, 840, 780, 720, 660, 600, 540] # 3080Ti
memFrequencylist = [9501,9251,5001,810,420]
MaxBatchSize=256

trtFile = "/home/model/vgg19.plan"

data_transform = {
    "test": transforms.Compose([transforms.Resize([224,224]),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


test_dataset = TinyImageNet("/home/data/imagenet/tiny-imagenet-200", train=False,transform=data_transform["test"])
print(test_dataset)
train_num = len(test_dataset)
print(train_num)
# #构建logger,builder,network
logger  = trt.Logger(trt.Logger.ERROR)
    
with open(trtFile, "rb") as f:
    engineString=f.read()
#反序列化构建tensorrt引擎
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engineString)
#构建执行上下文环境
context = engine.create_execution_context()

# context.profiler =MyProfiler()
def one_row(batch_size):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
    sum_time=0
    # sum_time1=0
    acc = 0
    total_num = 0
    total_round=0
    output=[]
    value=0.0
    _, stream = cudart.cudaStreamCreate() 
    value1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for xTest, yTest in test_loader:
        # print(xTest.numpy().shape)
        #这句非常重要！！！定义batch为动态维度,设置输入维度大小
        _,trtTimeStart=cudart.cudaEventCreate()
        _,trtTimeEnd=cudart.cudaEventCreate()
        cudart.cudaEventRecord(trtTimeStart,stream)
        # cudart.cudaStreamSynchronize(stream)
        # trtTimeStart = time.time()
        
        context.set_binding_shape(0, [batch_size, 3, nHeight, nWidth])
        inputH0 = np.ascontiguousarray(xTest.numpy().reshape(-1))
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
        # 完整一次推理
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaEventRecord(trtTimeEnd,stream)
        cudart.cudaEventSynchronize(trtTimeEnd)
        _,cost=cudart.cudaEventElapsedTime(trtTimeStart, trtTimeEnd)
        # outputH0=torch.max(torch.from_numpy(outputH0), dim=1)[1]
        # cudart.cudaStreamSynchronize(stream)
        # trtTimeEnd = time.time()
        cudart.cudaFree(inputD0)
        cudart.cudaFree(outputD0)
        cudart.cudaEventDestroy(trtTimeStart)
        cudart.cudaEventDestroy(trtTimeEnd)
        total_num+=len(xTest)
        if len(xTest)==batch_size:
            # t=(trtTimeEnd - trtTimeStart) * 1000
            sum_time+=cost
            total_round+=1
            if total_round == 100:
                break
        # sum_time1+=context.profiler.getTotalTimes()
        # if(len(outputH0)>len(yTest)):
        #     outputH0=outputH0[0:len(yTest)]
        # acc += torch.sum(outputH0 ==yTest).sum().item()
    
        # if n>1000:
        #     break
        # print("time: %fms acc: %f "%(t ,acc/n))
    value2 =  pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    value=value2-value1
    return (sum_time+0.0)/total_round,(value+0.0)/total_num
    print("avgtime: %.3fms"%())
    # print("avgtime: %.3fms"%(batch_size*sum_time1/n))
    print("avgenery: %.3fmJ"%(value/total_num))
    # print("Succeeded running model in TensorRT!")
if __name__=="__main__":
    # result_file=open("vgg19_result.txt","w",buffering = 0)
    logging.basicConfig(filename='vgg19_result.txt',filemode='w',level=logging.INFO)
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle,pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))
    pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
    time.sleep(1)
    for memFrequency in memFrequencylist:
        pynvml.nvmlDeviceSetMemoryLockedClocks(handle,memFrequency,memFrequency)
        for frequency in frequencylist:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle,frequency,frequency)
            time.sleep(1)
            for batch_size in batchsizelist:
                energy=0
                exectime=0
                logging.info("memory frequency:%d,frequency:%d,batch_size:%d\n"%(memFrequency,frequency,batch_size))
                # result_file.write("frequency:%d,batch_size:%d\n"%(frequency,batch_size))
                n=5
                for i in range(n):
                    tmp_energy,tmp_exectime=one_row(batch_size)
                    energy+=tmp_energy
                    exectime+=tmp_exectime
                    time.sleep(1)
                logging.info("avgtime: %.4fms\n"%(energy/n))
                logging.info("avgenery: %.4fmJ\n"%(exectime/n))
                print('memory frequency:%d,frequency:%d,batch_size:%d,avgtime: %.4fms,avgenery: %.4fmJ'%(memFrequency,frequency,batch_size,energy/n,exectime/n))
                # result_file.write("avgtime: %.4fms\n"%(energy/n))
                # result_file.write("avgenery: %.4fmJ\n"%(exectime/n))
    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle,pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))