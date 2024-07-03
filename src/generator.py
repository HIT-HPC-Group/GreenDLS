import socket
import os
import random
import numpy as np
import torch
from PIL import Image
import time
import json
from torchvision import transforms,datasets
import queue
import threading
import sys
sys.path.append(r"/home/data")
from imagenet import *
import glva
data_set_load = []

def initialization():
    global data_set_load
    print("Initialization Start...")
    ## 获取数据集
    data_transform = {
    "train": transforms.Compose([
                                transforms.Resize([224,224]),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_dataset = TinyImageNet("/home/data/imagenet/tiny-imagenet-200", train=False,transform=data_transform["train"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = 8, drop_last = True)
    for xTest, yTest in test_loader:
        xTest=xTest.numpy()
        data_set_load.append(xTest)


    print("Initialization End...")
    


# Simulate send time by introducing a random delay of at most this many seconds
max_item_delay_seconds = .0000000000001

# Simulate send time by introducing a random delay of at most this many seconds
# max_item_delay_seconds = 7e-4
# # How many items to send
# Do something representing a send, introducing a random delay
def do_one_item(numbers,item_count):
    i=0
    # max_item_delay_seconds=1.0/(numbers+80)
    start=time.time()
    for t in range(numbers):  
        data = data_set_load[i]
        i=(i+1)%item_count
        rtime = time.time()
        glva.gl_requestQ.append([data,rtime]) ##请求队列
        glva.gl_req_count[0]+=1
        # 请求入队列
        # Compute how much time we've spent so far
        time_spent = time.time() - start
        # Compute how much time we want to have spent so far based on the desired send rate
        should_time = (t + 1) / numbers
        # If we're going too fast, wait just long enough to get us back on track
        if should_time > time_spent:
            time.sleep(should_time - time_spent)
### 本地模式来模拟请求
def get_request_local(ite, datalen):
    # Record the starting time
    start_time = time.time()
    sendnum=0
    item_count = len(data_set_load)
    i=0
    print("item_count :{}\n".format(item_count))
    while True:
        glva.sends_per_second=glva.sends_speed_list[i]
        i=(i+1)%len(glva.sends_speed_list)
        glva.sendspeed[0]=glva.sends_per_second
        t1 = time.time()
        do_one_item(glva.sends_per_second,item_count)
        sendnum+=glva.sends_per_second
        # Compute how much time we've spent so far
        time_spent = time.time() - t1
        print("time spend : {}".format(time_spent))
        time_s = time.time() - start_time
        print("current speed {} ,Sent {} items in {} seconds (averagee {} items per second)".format(glva.sends_per_second,sendnum, time_s, sendnum / time_s))
    
class requestThread_Local(threading.Thread):
    def __init__(self,ite,datalen):
        super(requestThread_Local,self).__init__()
        self.ite = ite
        self.datalen = datalen
    def run(self):
        get_request_local(self.ite, self.datalen)


