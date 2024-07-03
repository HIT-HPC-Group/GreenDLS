import os
import numpy as np
import torch
import time
import threading
import math
from rlmodel.DuelingDQN import *
from rlmodel.rl_utils import *
import tensorrt as trt
import matplotlib.pyplot as plt
from cuda import cudart
import pynvml
import glva
from scipy.special import expit

def load_model(trtFile):
    logger = trt.Logger(trt.Logger.VERBOSE)
    with open(trtFile, "rb") as f:
        engineString = f.read()
    # 反序列化构建tensorrt引擎
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engineString)
    # 构建执行上下文环境
    context = engine.create_execution_context()
    return engine,context


def do_predict(img,engine,context,batch_size):
    
    value1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    context.set_binding_shape(0, [batch_size, 3, glva.nHeight, glva.nWidth])
    _, stream = cudart.cudaStreamCreate()
    inputH0 = np.ascontiguousarray(img.reshape(-1))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)  # 推理数据上传时间
    context.execute_async_v2([int(inputD0), int(outputD0)], stream) #推理执行时间
    #print("推理执行中..............................................................................")
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)  #推理数据返回时间
    outputH0 = torch.max(torch.from_numpy(outputH0), dim=1)[1]
    # result_txt.write("推理结果: "+str(outputH0)+"\n")
    cudart.cudaStreamSynchronize(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    cudart.cudaStreamDestroy(stream)
    value2 = pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    slo = time.time()
    return slo, value1, value2

def eaisDecision():
    speed=glva.sendspeed[0]
    b=1
    f=2100
    frequencys=['540','660','780','900','1020','1140','1260','1380','1500','1620','1740','1860','1980','2100']
    energy=math.inf
    for fre in frequencys:
        #300 1500
        #400
        max_b=int(glva.sendspeed[0]*(glva.SLO/1200-(time.time()-glva.gl_requestQ[0][1]))+len(glva.gl_requestQ))
        tmp_b=1
        for bathsize in range(max_b,-1,-4):
            costtime=glva.time_predict_model[fre][0]*bathsize+glva.time_predict_model[fre][1]
            waittime=1200*(time.time()-glva.gl_requestQ[0][1]+(bathsize-len(glva.gl_requestQ))/glva.sendspeed[0])
            if waittime+costtime<glva.SLO:
                tmp_b=bathsize
                break
        tmp=glva.energy_predict_model.predict([[tmp_b,fre]])[0]
        if tmp<energy:
            energy=tmp
            b=tmp_b
            f=fre      
    b=min(b,256)
    b=max(b,1)          
    return b,int(f)
def eaisSecondDecision(b):
    speed=glva.sendspeed[0]
    f=1597
    frequencys=[517, 577, 637, 697, 757, 817, 877, 937, 997, 1057, 1117, 1177, 1237, 1297, 1357, 1417, 1477, 1537,1597]
    energy=math.inf
    for fre in frequencys:
        tmp=glva.energy_predict_model.predict([[b,fre]])[0]
        if tmp<energy:
            energy=tmp
            f=fre                
    return b,f
def eais(engine,context):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=1
    total_energy=0
    time_total=0
    total_energy_predict = 0
    num_step=0
    start_time=time.time()
    inital_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    nums = 0
    while num_sum<glva.gl_max_count:
        tt1=time.time()
        batchsize,gpu_fre=eaisDecision()
        tt2=time.time()
        glva.result_txt.write("eais决策时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
        glva.result_txt.write("本次调节的batchsize: "+str(batchsize)+" frequency: "+str(gpu_fre)+"\n")
        batch_energy = 0
        batch_times = []
        batch_inputs = []  
        while len(glva.gl_requestQ)<=batchsize:
            continue
        pynvml.nvmlDeviceSetGpuLockedClocks(glva.handle,210,gpu_fre)
        if len(glva.gl_requestQ)>=batchsize:
            tt1 = time.time()
            for j in range(batchsize):
                tmp = glva.gl_requestQ.popleft() 
                batch_inputs.append(tmp[0])
                batch_times.append(tmp[1])
            batch_inputs=np.array(batch_inputs)  
            tt2 = time.time()
            glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
            glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
            t1 = time.time()
            task=glva.pool.submit(do_predict,batch_inputs,engine,context,batchsize)
            while task.done()==False:
                power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                batch_energy+=power_gpu*2
                power_gpu=power_gpu/350
                delay_mark = time.time()    
                time.sleep(0.001)
            slo, value1, value2 = task.result()
            t2 = time.time()
            num_step+=1
            glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str((t2 - t1) * 1000)+"\n")
            time_total += (t2 - t1) 
            glva.result_txt.write("=====================单批次能耗(单位:J): "+str(batch_energy)+"\n")
            total_energy+=batch_energy
            glva.result_txt.write("=====================单个推理功耗："+str(batch_energy/batchsize)+"\n")
            batch_slo = [(slo-batch_times[n])*1000 for n in range(batchsize)]
            num_overslo+=len([x for x in batch_slo if x>glva.SLO])
            num_sum+=len(batch_slo)
            nums+=1
            glva.result_txt.write("=====================当前批次为："+str(nums)+"\n")
            glva.result_txt.write("=====================本轮调节批大小："+str(batchsize)+"\n")
            glva.result_txt.write("=====================本轮推理时延(单位:ms)："+str(batch_slo)+"\n")  
            glva.result_txt.write("=====================累计超出SLO个数："+str(num_overslo)+"\n")  
            glva.result_txt.write("=====================总计完成推理的个数："+str(num_sum)+"\n")
            glva.result_txt.write("=============================平均批大小："+str(num_sum/num_step)+"\n")
            glva.result_txt.write("=====================累计总能量消耗（单位：J）："+str(total_energy)+"\n")  
            glva.result_txt.write("=====================累计推理总时长（单位：s）："+str(time_total)+"\n")
            end=time.time()-start  
            glva.result_txt.write("=====================累计运行总时长（单位：s）："+str(end)+"\n")
            energy = batchsize*glva.energy_predict_model.predict([[batchsize,pynvml.nvmlDeviceGetClockInfo(glva.handle,0)]])[0]
            total_energy_predict +=energy
            glva.result_txt.write("=====================预测的总能耗推理功耗："+str(total_energy_predict)+"\n")
            i = i + len(batch_slo)        
    finish_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    end_time=time.time()-start_time
    base_energy=29*end_time*1000
    glva.result_txt.write("=====================推理总功耗："+str(finish_value-inital_value-base_energy)+" mJ \n")
def clipper(engine,context):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=1
    total_energy=0
    time_total=0
    total_energy_predict = 0
    num_step=0
    start_time=time.time()
    inital_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    while num_sum<glva.gl_max_count:
        batch_energy = 0
        batch_times = []
        batch_inputs = []       
        while len(glva.gl_requestQ)<=batchsize:
            continue
        if len(glva.gl_requestQ)>=batchsize:
            tt1 = time.time()
            for j in range(batchsize):
                tmp = glva.gl_requestQ.popleft() 
                batch_inputs.append(tmp[0])
                batch_times.append(tmp[1])
            batch_inputs=np.array(batch_inputs)  
            tt2 = time.time()
            glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
            glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
            t1 = time.time()
            task=glva.pool.submit(do_predict,batch_inputs,engine,context,batchsize)
            while task.done()==False:
                power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                batch_energy+=power_gpu*2
                power_gpu=power_gpu/350
                
                time.sleep(0.001)
            slo, value1, value2 = task.result()
            t2 = time.time()
            num_step+=1
            glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str((t2 - t1) * 1000)+"\n")
            time_total += (t2 - t1) 
            glva.result_txt.write("=====================单批次能耗(单位:J): "+str(batch_energy)+"\n")
            total_energy+=batch_energy
            glva.result_txt.write("=====================单个推理功耗："+str(batch_energy/batchsize)+"\n")
            batch_slo = [(slo-batch_times[n])*1000 for n in range(batchsize)]
            num_overslo+=len([x for x in batch_slo if x>glva.SLO])
            num_sum+=len(batch_slo)
            glva.result_txt.write("=====================本轮调节批大小："+str(batchsize)+"\n")
            glva.result_txt.write("=====================本轮推理时延(单位:ms)："+str(batch_slo)+"\n")  
            glva.result_txt.write("=====================累计超出SLO个数："+str(num_overslo)+"\n")  
            glva.result_txt.write("=====================总计完成推理的个数："+str(num_sum)+"\n")
            glva.result_txt.write("=============================平均批大小："+str(num_sum/num_step)+"\n")
            glva.result_txt.write("=====================累计总能量消耗（单位：J）："+str(total_energy)+"\n")  
            glva.result_txt.write("=====================累计推理总时长（单位：s）："+str(time_total)+"\n")
            end=time.time()-start  
            glva.result_txt.write("=====================累计运行总时长（单位：s）："+str(end)+"\n")
            i = i + len(batch_slo)
            if batch_slo[0]>glva.SLO:
                batchsize=int(0.5*batchsize)
                batchsize= batchsize if batchsize>=1 else 1
            elif batch_slo[0]>0.95*glva.SLO:
                batchsize-=2 if batchsize>=1 else 1
            else:
                batchsize+=1
    finish_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    end_time=time.time()-start_time
    base_energy=29*end_time*1000
    glva.result_txt.write("=====================推理总功耗："+str(finish_value-inital_value-base_energy)+" mJ \n")
def predict_rl(engine,context,RL):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=glva.batch_size[0]
    total_reward=0
    total_energy=0
    time_total=0
    total_energy_predict=0
    num_step=0
    obeservation=[0.1,0,0]
    obeservation.append(glva.sendspeed[0]/1000)
    last_gl_len=0
    start_time=time.time()
    inital_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    while num_sum<glva.gl_max_count:
        batch_energy = 0
        batch_times = []
        batch_inputs = []
        rltime=time.time()
        action = RL.choose_action(obeservation,is_trained=False)
        rltime=time.time()-rltime
        batchsize=glva.action_GPU[action][2]
        gpu_fre = glva.action_GPU[action][1]
        mem_fre = glva.action_GPU[action][0]
        done = 1
        waittime=1200*(time.time()-glva.gl_requestQ[0][1]+(batchsize-len(glva.gl_requestQ))/glva.sendspeed[0])
        glva.result_txt.write("最长等待时间(单位:ms): "+str(waittime)+"\n")
        if waittime >=glva.SLO:
            batchsize=len(glva.gl_requestQ)
        elif waittime<glva.SLO and waittime >=glva.SLO*0.92:
            batchsize=int(batchsize*0.85)
        glva.result_txt.write("**********调节频率的大小："+str(gpu_fre)+"\n")
        glva.result_txt.write("**********调节内存频率的大小："+str(mem_fre)+"\n")
        pynvml.nvmlDeviceSetMemoryLockedClocks(glva.handle,405,mem_fre)
        pynvml.nvmlDeviceSetGpuLockedClocks(glva.handle,210,gpu_fre)
        while len(glva.gl_requestQ)<=batchsize:
            continue
        if len(glva.gl_requestQ)>=batchsize:
            tt1 = time.time()
            for j in range(batchsize):
                tmp = glva.gl_requestQ.popleft()
                batch_inputs.append(tmp[0])
                batch_times.append(tmp[1])
            batch_inputs=np.array(batch_inputs)  
            tt2 = time.time()
            longest_wait_time=(tt2 - batch_times[0])*1000
            glva.result_txt.write("最长等待时间(单位:ms): "+str(longest_wait_time)+"\n")
            
            glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
            glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
            batch_energy=0
            sum_gpu_state=[0]
            rounds=0
            t1 = time.time()
            task=glva.pool.submit(do_predict,batch_inputs,engine,context,batchsize)
            while task.done()==False:
                power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                batch_energy+=power_gpu*2
                power_gpu=power_gpu/350
                gpu_state = [power_gpu]  # wym modify
                sum_gpu_state=[x+y for x, y in zip(sum_gpu_state,gpu_state)]
                rounds+=1
                time.sleep(0.001)
            t2 = time.time()
            slo, value1, value2 = task.result()
            pynvml.nvmlDeviceSetMemoryLockedClocks(glva.handle,405,405)
            num_step+=1
            batch_time=(t2 - tt1) * 1000
            glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str(batch_time)+"\n")
            
            time_total += (t2 - t1) 
            sum_gpu_state=[round(x/rounds,3) for x in sum_gpu_state]
            glva.result_txt.write("=====================单批次能耗(单位:mJ): "+str(batch_energy)+"\n")
            total_energy+=batch_energy
            glva.result_txt.write("=====================单个推理功耗："+str(batch_energy/batchsize)+"\n")
            glva.result_txt.write("=====================单个推理功耗(-29w)："+str((batch_energy-(t2-t1)*29*1000)/batchsize)+"\n")
            batch_slo = [(slo-batch_times[n])*1000 for n in range(batchsize)]
            num_overslo+=len([x for x in batch_slo if x>glva.SLO])
            num_sum+=len(batch_slo)
            glva.result_txt.write("=====================本轮调节批大小："+str(batchsize)+"\n")
            glva.result_txt.write("=====================本轮推理时延(单位:ms)："+str(batch_slo)+"\n")  
            glva.result_txt.write("=====================累计超出SLO个数："+str(num_overslo)+"\n")  
            glva.result_txt.write("=====================总计完成推理的个数："+str(num_sum)+"\n")
            glva.result_txt.write("=============================平均批大小："+str(num_sum/num_step)+"\n")
            glva.result_txt.write("=====================累计总能量消耗（单位：mJ）："+str(total_energy)+"\n") 
            glva.result_txt.write("=====================累计总能量消耗（单位：mJ）(-29w)："+str(total_energy-time_total*29*1000)+"\n")   
            glva.result_txt.write("=====================累计推理总时长（单位：s）："+str(time_total)+"\n")
            end=time.time()-start  
            glva.result_txt.write("=====================累计运行总时长（单位：s）："+str(end)+"\n")
            i = i + len(batch_slo)
            time1=longest_wait_time
            longest_wait_time=1 if longest_wait_time>glva.SLO else longest_wait_time/glva.SLO
            batch_time=1 if batch_time>glva.SLO else batch_time/glva.SLO
            sum_gpu_state.append(round(longest_wait_time,3))
            sum_gpu_state.append(round(batch_time,3))
            currentspeed=glva.sendspeed[0]
            currentspeed=currentspeed/1000
            sum_gpu_state.append(round(currentspeed,3))
            longest_wait_time=time1
            glva.result_txt.write("=====================当前状态："+str(sum_gpu_state)+"\n")
            obeservation_=sum_gpu_state  
            reward=0
            delay=batch_slo[0]
            energy=glva.energy_predict_model.predict([[mem_fre,gpu_fre,batchsize]])[0]
            glva.result_txt.write("=====================预测的单个推理功耗："+str(energy)+"\n")
            total_energy_predict+=energy*batchsize
            glva.result_txt.write("=====================预测的总能耗推理功耗："+str(total_energy_predict)+"\n")
            k=150
            
            if delay<=glva.SLO:
                reward=k*delay/energy
            else:
                reward=glva.SLO-delay
            
            obeservation=obeservation_
          
    finish_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    end_time=time.time()-start_time
    base_energy=29*end_time*1000
    glva.result_txt.write("=====================推理总功耗："+str(finish_value-inital_value-base_energy)+" mJ \n")
def train(engine,context,RL):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=glva.batch_size[0]
    total_reward=0
    total_energy=0
    time_total=0
    last_gl_len=0
    num_step=0
    obeservation=[0.1,0,0]
    obeservation.append(glva.sendspeed[0]/1000)
    epochs=500
    max_q_value=0
    finish=True
    for epoch in range(epochs):
        total_reward=0
        print("epoch : ",epoch)
        for step in range(int(STEPS/epochs)):
            finish=True
            while finish:
                done = 1
                batch_energy = 0
                batch_times = []
                batch_inputs = []
                rltime=time.time()
                action = RL.choose_action(obeservation,is_trained=True)
                batchsize=glva.action_GPU[action][2]
                gpu_fre = glva.action_GPU[action][1]
                mem_fre = glva.action_GPU[action][0]
                max_q_value = RL.get_QValue(obeservation) * 0.005 + max_q_value * 0.995  # 平滑处理
                glva.result_txt.write("-------------Q Value: "+str(max_q_value)+"\n")
                rltime=time.time()-rltime
                glva.result_txt.write("**********调节频率的大小："+str(gpu_fre)+"\n")
                glva.result_txt.write("**********调节内存频率的大小："+str(mem_fre)+"\n")
                pynvml.nvmlDeviceSetMemoryLockedClocks(glva.handle,420,mem_fre)
                pynvml.nvmlDeviceSetGpuLockedClocks(glva.handle,210,gpu_fre)
                while len(glva.gl_requestQ)<=batchsize:
                    continue
                if len(glva.gl_requestQ)>=batchsize:
                    finish=False
                    tt1 = time.time()
                    for j in range(batchsize):
                        tmp = glva.gl_requestQ.popleft() 
                        batch_inputs.append(tmp[0])
                        batch_times.append(tmp[1])
                    batch_inputs=np.array(batch_inputs)  
                    tt2 = time.time()
                    longest_wait_time=(tt2 - batch_times[0])*1000
                    glva.result_txt.write("最长等待时间(单位:ms): "+str(longest_wait_time)+"\n")
                    wait_time=longest_wait_time
                    glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
                    glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
                    batch_energy=0
                    sum_gpu_state=[0]
                    rounds=0
                    t1 = time.time()
                    task=glva.pool.submit(do_predict,batch_inputs,engine,context,batchsize)
                    while task.done()==False:
                        power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                        batch_energy+=power_gpu*2
                        power_gpu=power_gpu/350
                        gpu_state = [power_gpu]  
                        sum_gpu_state=[x+y for x, y in zip(sum_gpu_state,gpu_state)]
                        rounds+=1
                        time.sleep(0.001)
                    t2 = time.time()
                    slo, value1, value2 = task.result()
                    num_step+=1
                    batch_time=(t2 - tt1) * 1000
                    glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str(batch_time)+"\n")
                    time_total += (t2 - t1) 
                    sum_gpu_state=[round(x/rounds,3) for x in sum_gpu_state]
                    glva.result_txt.write("=====================单批次能耗(单位:mJ): "+str(batch_energy)+"\n")
                    total_energy+=batch_energy
                    glva.result_txt.write("=====================单个推理功耗："+str(batch_energy/batchsize)+"\n")
                    batch_slo = [(slo-batch_times[n])*1000 for n in range(batchsize)]
                    num_overslo+=len([x for x in batch_slo if x>glva.SLO])
                    num_sum+=len(batch_slo)
                    glva.result_txt.write("=====================本轮调节批大小："+str(batchsize)+"\n")
                    glva.result_txt.write("=====================本轮推理时延(单位:ms)："+str(batch_slo)+"\n")  
                    glva.result_txt.write("=====================累计超出SLO个数："+str(num_overslo)+"\n")  
                    glva.result_txt.write("=====================总计完成推理的个数："+str(num_sum)+"\n")
                    glva.result_txt.write("=============================平均批大小："+str(num_sum/num_step)+"\n")
                    glva.result_txt.write("=====================累计总能量消耗（单位：J）："+str(total_energy)+"\n")  
                    glva.result_txt.write("=====================累计推理总时长（单位：s）："+str(time_total)+"\n")
                    end=time.time()-start  
                    glva.result_txt.write("=====================累计运行总时长（单位：s）："+str(end)+"\n")
                    i = i + len(batch_slo)
                    longest_wait_time=1 if longest_wait_time>glva.SLO else longest_wait_time/glva.SLO
                    batch_time=1 if batch_time>glva.SLO else batch_time/glva.SLO
                    sum_gpu_state.append(round(longest_wait_time,3))
                    sum_gpu_state.append(round(batch_time,3))
                    longest_wait_time=wait_time
                    currentspeed=glva.sendspeed[0]
                    currentspeed=currentspeed/1000
                    sum_gpu_state.append(round(currentspeed,3))
                    glva.result_txt.write("=====================当前状态："+str(sum_gpu_state)+"\n")
                    obeservation_=sum_gpu_state
                    reward=0
                    delay=batch_slo[0]
                    
                    energy=glva.energy_predict_model.predict([[mem_fre,gpu_fre,batchsize]])[0]
                    glva.result_txt.write("=====================预测的单个推理功耗："+str(energy)+"\n")
                    k=150
                    if delay<=0.92*glva.SLO:
                        reward=k*delay/energy
                    else:
                        done = 0
                        reward=0.92*glva.SLO-delay
                    glva.result_txt.write('***********current reward : '+str(reward)+"\n")
                    total_reward+=reward
                    glva.result_txt.write('***********done : '+str(done)+"\n")
                    RL.remember(obeservation, action, reward, obeservation_, done)
                    loss=RL.learn()
                    glva.result_txt.write('***********current loss : '+str(loss)+"\n")

                    obeservation = obeservation_
                if num_step%50==0:
                    print("current step: ",num_step) 
                if num_step==STEPS:
                    break     
            if num_step==STEPS:
                break               
        glva.result_txt.write('***********total reward : '+str(total_reward)+"\n") 
        if num_step==STEPS:
            break         

       

class inferenceThread(threading.Thread):
    def __init__(self,ite,datalen,RL):
        super(inferenceThread,self).__init__()
        self.engine = ite
        self.context = datalen
        self.RL=RL
        self.RUN_MODE=glva.RUN_MODE
    def run(self):
        if self.RUN_MODE ==0:
            train(self.engine,self.context,self.RL)
            torch.save(self.RL,glva.rlmodel)
        elif self.RUN_MODE==1:
            predict_rl(self.engine,self.context,self.RL)
            # torch.save(self.RL,"vgg19/train.pth")
        elif self.RUN_MODE==2:
            clipper(self.engine,self.context)
        elif self.RUN_MODE==3:
            eais(self.engine,self.context)
        
        glva.result_txt.close()
        pynvml.nvmlShutdown()
        
        