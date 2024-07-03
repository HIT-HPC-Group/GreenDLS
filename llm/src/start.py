import socket
import os
import threading
import numpy as np
import pynvml
# from monitor import get_power
import glva
from rlmodel.DuelingDQN import *
from rlmodel.rl_utils import *
# import eais
import scheduler
import generator
import torch
import time
import sys
import argparse

def read_argparse():    
    parser = argparse.ArgumentParser(description='inital global informations')
    parser.add_argument('--SLO', dest='SLO',default=200, type=int, help='SLO :{}'.format(200,300,400,500))
    parser.add_argument('--sends_per_second', dest='sends_per_second',default=180, type=int, help='Client request rate')
    parser.add_argument('--sends_speed_file', dest='sends_speed_file',default="dynamic.txt" ,type=str, help='List of dynamic request rates')
    parser.add_argument('--RUN_MODE', dest='RUN_MODE',default=1 ,type=int, help='0:TRAIN 1:TEST 2:Clipper 3:EAIS')
    parser.add_argument('--save_path', dest='save_path',default="./vgg19/dynamic" ,type=str, help='Scheduling record saving address')
    parser.add_argument('--trtFile', dest='trtFile',default="/home/model/vgg19.plan" ,type=str, help='tensorrt file')
    parser.add_argument('--rlmodel', dest='rlmodel',default="vgg19/train.pth" ,type=str, help='RL model save path')
    args=parser.parse_args()
    glva.SLO=args.SLO
    glva.sends_per_second=args.sends_per_second
    glva.sends_speed_file=args.sends_speed_file
    glva.RUN_MODE=args.RUN_MODE
    glva.save_path=args.save_path
    glva.trtFile=args.trtFile
    glva.rlmodel=args.rlmodel
device = torch.device("cpu")
if __name__ == '__main__':
    seed= 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    read_argparse()

    os.system("nvidia-smi -rgc")
    os.system("nvidia-smi -pl 250")
    print("actions {}".format(len(glva.action_GPU)))
    pynvml.nvmlInit()
    generator.initialization()
    model = 0
    tokenizer = 0
    if glva.RUN_MODE > 0 :
        model,tokenizer=scheduler.load_model(glva.trtFile)
        scheduler.do_predict(model,tokenizer,["讲一个关于能说话会下棋的机器人的故事"])
    if glva.RUN_MODE == 0:
        agent =DuelingDQN(alpha=0.0005, state_dim=4, action_dim=len(glva.action_GPU),
                        fc1_dim=32, fc2_dim=32,device=device, gamma=0.99, tau=0.05, epsilon=1.0,
                        eps_end=0.1, eps_dec=8e-5, max_size=100000, batch_size=64)

        obeservation=[0.1,0,0,0]
        action = agent.choose_action(obeservation)
        t1 = generator.requestThread_Local(0,10000)
        t2 = scheduler.inferenceThread(model,tokenizer,agent)
        t1.start()
        t2.start()
        t2.join()
    elif glva.RUN_MODE==1:
        print(glva.rlmodel)
        agent=torch.load(glva.rlmodel) 
        agent.epsilon=0.00
        agent.eps_end=0.00
        obeservation=[0.1,0,0,0]
        action = agent.choose_action(obeservation,is_trained=True)
        t1 = generator.requestThread_Local(0,10000)
        t2 = scheduler.inferenceThread(model,tokenizer,agent)

        t1.start()
        t2.start()
        t2.join()
    else:
        t1 = generator.requestThread_Local(0,10000)
        t2 = scheduler.inferenceThread(model,tokenizer,None)
        t1.start()
        t2.start()
        t2.join()
        
    os._exit(0)

    