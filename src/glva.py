import queue
from collections import deque
import threading
import pynvml
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import joblib
###threading config
# gl_requestQ = queue.Queue(maxsize=0) ##请求队列
gl_requestQ = deque() ##请求队列
gl_req_count = [0] ##请求的数量，用于测量请求到来的速度
nHeight = 224
nWidth = 224
gl_mutex = threading.Lock()
SLO=200
# Send this many items per second
sends_per_second =270
sendspeed=[sends_per_second]

sends_speed_file = ""
# sends_speed_file="wiki.txt"
# sends_speed_file="sogouQ.txt"
# sends_speed_file="dynamic.txt"
# sends_speed_file="trainspeed.txt"
sends_speed_list = None # start.py传参后再打开
gl_max_count = None
# sends_speed_list=np.loadtxt(sends_speed_file,delimiter=',',dtype=int)
# gl_max_count=sends_speed_list.sum()
batch_size=[4]
# over_slo_=[0.0]

def get_eais_(file):
    result_txt=open(file,"r")
    lines = result_txt.readlines()
    eias_result={}
    for line in lines:
        line=line.replace("\n", "")
        frequency=line.split(':')[0]
        index=line.split(':')[1]
        index=index.split(',')
        index=[float(x) for x in index]
        eias_result[frequency]=index
    return eias_result
eais_num_step=0
eais_num_overslo=0
eais_num_sum=0
eais_lock=threading.Lock()
#0:TRAIN 1:TEST 2:Clipper 3:EAIS
RUN_MODE=1
# save_path="./densenet_result/SLO{}_Speed{}".format(SLO,sends_per_second)
# energy_predict_model=joblib.load("./resnet50/resnet50_energy.pkl")
# time_predict_model=get_eais_("./resnet50/resnet50_time_batchsize.txt")
energy_predict_model=joblib.load("./densenet/densenet201_energy.pkl")
time_predict_model=get_eais_("./densenet/densenet_time_batchsize.txt")
# energy_predict_model=joblib.load("./vgg19/vgg19_energy.pkl")
# time_predict_model=get_eais_("./vgg19/vgg19_time_batchsize.txt")
###
#mobilenetv2、alexnet 800
#shufflenet、resnet50 :640 1024
#vgg19 360、160
#densenet210 360
#convnext 320
save_path=""
trtFile=""
rlmodel=""
# save_path="./resnet50/clipperdynamic"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("./resnet50/clipperdynamicstatus.txt","w")

# save_path="./resnet50/clipperwiki"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("resnet50/clipperwikistatus.txt","w")

# save_path="./resnet50/wiki"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("resnet50/wikistatus.txt","w")

# save_path="./resnet50/clippersougouQ"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("resnet50/clippersougouQstatus.txt","w")

# save_path="./resnet50/sougouQ"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("resnet50/sougouQstatus.txt","w")

# save_path="./resnet50/dynamic"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("resnet50/dynamicstatus.txt","w")

# save_path="./resnet50/eaisdynamic"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("./resnet50/eaisdynamicstatus.txt","w")

# save_path="./resnet50/eaissougouQ"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("./resnet50/eaissougouQstatus.txt","w")

# save_path="./resnet50/eaiswiki"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# gpu_information=open("./resnet50/eaiswikistatus.txt","w")

# save_path="./resnet50/train"
# trtFile = "/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"


# save_path="./vgg19/clipperdynamic"
# trtFile = "/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("./vgg19/clipperdynamicstatus.txt","w")

# save_path="./vgg19/dynamic"
# trtFile = "/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/dynamicstatus.txt","w")

# save_path="./vgg19/clipperwiki"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/clipperwikistatus.txt","w")

# save_path="./vgg19/eaiswiki"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/eaiswikistatus.txt","w")

# save_path="./vgg19/eaisdynamic"
# trtFile = "/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("./vgg19/eaisdynamicstatus.txt","w")


# save_path="./vgg19/wiki"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/wikistatus.txt","w")

# save_path="./vgg19/eaissougouQ"
# trtFile = "/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/eaissougouQstatus.txt","w")

# save_path="./vgg19/sougouQ"
# trtFile = "/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/sougouQstatus.txt","w")

# save_path="./vgg19/clippersougouQ"
# trtFile = "/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("vgg19/clippersougouQstatus.txt","w")

# save_path="./vgg19/train"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# gpu_information=open("./vgg19/status.txt","w")

# save_path="./densenet/eaiswiki"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/clipperwiki"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/wiki"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/speeds"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/sougouQ"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/eaissougouQ"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/clippersougouQ"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/clipperdynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/eaisdynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"


# save_path="./densenet/dynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"

# save_path="./densenet/train"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# gpu_information=open("./densenet/status.txt","w")

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#result_txt=open(save_path+".txt","w")
result_txt = None # start.py传参后再打开

# batchsizes=[128,64,32,16,8,4,2,1]
# batchsizes=[1,2,4,8,16,32,64,128]
# batchsizes=[1,2,4,8,16]
batchsizes=[x for x in range(4,65,4)]
batchsizes.append(1)
batchsizes.append(2)
# batchsizes[0]=1
memFrequencylist = [9501,9251,5001,810,405]

def get_actions():
    max_freq = 2100
    min_freq = 540
    clock = min_freq
    GPU = []
    action_GPU=[]
    while clock <= max_freq:
        GPU.append(clock)
        clock = clock+60
    print(GPU)
    gpu_to_bucket = {GPU[i]: i for i in range(len(GPU))}
    for memFrequency in memFrequencylist:
        for gpu in GPU:
            for batch in batchsizes:
                action_GPU.append([memFrequency,gpu,batch])
    return action_GPU
action_GPU=get_actions()

max_clock = 2100
min_clock = 210
clock=min_clock
CLOCKS_GPU =[]
while clock <= max_clock:
    CLOCKS_GPU.append(clock)
    clock = clock + 60
# print(CLOCKS_GPU)
pool = ThreadPoolExecutor(max_workers=1)
