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
SLO=10000
# Send this many items per second
sends_per_second =270
sendspeed=[sends_per_second]
# sends_speed_file="speeds.txt"
# sends_speed_file="wiki.txt"
# sends_speed_file="sogouQ.txt"
sends_speed_file="dynamic.txt"
# sends_speed_file="trainspeed.txt"
sends_speed_list=np.loadtxt(sends_speed_file,delimiter=',',dtype=int)
gl_max_count=sends_speed_list.sum()
batch_size=[4]
over_slo_=[0.0]

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
RUN_MODE=2

energy_predict_model=joblib.load("./qwen-int4/qwen7B-INT4_energy.pkl")
power_predict_model=joblib.load("./qwen-int4/qwen7B-INT4_power.pkl")
times_predict_model=joblib.load("./qwen-int4/qwen7B-INT4_time.pkl")
time_predict_model=get_eais_("./qwen-int4/qwen7B-INT4_time_batchsize.txt")

# save_path="./qwen-int4/eaisdynamic"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"

# save_path="./qwen-int4/speeds"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"

# save_path="./qwen-int4/train"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"

save_path="./qwen-int4/clipperdynamic"
trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
rlmodel="qwen-int4/train.pth"

# save_path="./qwen-int4/eaisdynamic"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"

# save_path="./qwen-int4/dynamic"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
result_txt=open(save_path+".txt","w")
batchsizes=[x for x in range(4,37,4)]
batchsizes.append(1)
batchsizes.append(2)

def get_actions():
    gpu_limit = 1590
    max_freq = 1597
    min_freq = 517
    clock = min_freq
    GPU = []
    action_GPU=[]
    while clock < max_freq:
        GPU.append(clock)
        clock = clock+60


    print(GPU)
    gpu_to_bucket = {GPU[i]: i for i in range(len(GPU))}

    for batch in batchsizes:
        for gpu in GPU:
            action_GPU.append([batch,gpu])
    return action_GPU
action_GPU=get_actions()

# 强化学习 State：(1)GPU频率，(2)GPU利用率，(3)显存利用率，(4)实时功耗，(5)实时温度，
GPU_LABELS = (
               'UTIL_GPU'
              , 'UTIL_MEM'
              , 'POWER'
              )
MINS = { 'UTIL_GPU': 0, 'UTIL_MEM': 0, 'POWER': 25}
MAXS = { 'UTIL_GPU': 100, 'UTIL_MEM': 100, 'POWER': 250}
BUCKETS = { 'UTIL_GPU': 100, 'UTIL_MEM': 100, 'POWER': 75}
gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)
gpu_all_mins = np.array([MINS[k] for k in GPU_LABELS], dtype=np.double)
gpu_all_maxs = np.array([MAXS[k] for k in GPU_LABELS], dtype=np.double)
gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)
#划分档位，共有BUCKETS个数个档
gpu_widths = np.divide(np.array(gpu_all_maxs) - np.array(gpu_all_mins), gpu_num_buckets)  # divide /
#gpu frequency as one of state
max_clock = 1597
min_clock = 135
clock=max_clock
CLOCKS_GPU =[]
while clock > min_clock:
    CLOCKS_GPU.append(clock)
    clock = clock-7
    CLOCKS_GPU.append(clock)
    clock = clock - 8
clock_gpu_bucket = { CLOCKS_GPU[i]: i for i in range(len(CLOCKS_GPU))}
pool = ThreadPoolExecutor(max_workers=1)
eais_pool = ThreadPoolExecutor(max_workers=2)
