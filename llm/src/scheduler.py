import os
import numpy as np
import torch
import time
import threading
import math
from rlmodel.DuelingDQN import *
from rlmodel.rl_utils import *
from cuda import cudart
import pynvml
import glva
import sys
from scipy.special import expit
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
model_path = "/home/model/qwen/Qwen-7B-Chat-Int4"
sys.path.append(model_path)
# model_path = ""
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

def load_model(model_path):
    # 可选的模型包括: "Qwen/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,pad_token='<|extra_0|>',eos_token='<|endoftext|>',padding_side='left')
    # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
    model = AutoModelForCausalLM.from_pretrained(model_path,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attn=False).eval()
    # 可指定不同的生成长度、top_p等相关超参
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True,pad_token_id=tokenizer.pad_token_id)
    return model,tokenizer

def do_predict(model,tokenizer,all_raw_text):
    value1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            tokenizer,
            q,
            system="You are a helpful assistant.",
            max_window_size=model.generation_config.max_window_size,
            chat_format=model.generation_config.chat_format,
        )
        batch_raw_text.append(raw_text)
    batch_input_ids = tokenizer(batch_raw_text, padding='longest')
    batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
    batch_out_ids = model.generate(
        batch_input_ids,
        return_dict_in_generate=False,
        generation_config=model.generation_config
    )
    value2 = pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    slo = time.time()
    return slo, value1, value2



def eaisDecision():
    speed=glva.sendspeed[0]
    b=1
    f=1597
    frequencys=['637','757','877','997','1117','1237','1357','1477','1597']
    energy=math.inf
    for fre in frequencys:
        max_b=min(int(glva.sendspeed[0]*(glva.SLO/1000-(time.time()-glva.gl_requestQ[0][1]))+len(glva.gl_requestQ)),32)
        tmp_b=1
        for bathsize in range(max_b,-1,-1):
            costtime=(glva.time_predict_model[fre][0]*bathsize+glva.time_predict_model[fre][1])*1000
            waittime=1200*(time.time()-glva.gl_requestQ[0][1]+(bathsize-len(glva.gl_requestQ))/glva.sendspeed[0])
            if waittime+costtime<0.9*glva.SLO:
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
    frequencys=[997,1117,1237,1357,1477,1597]
    energy=math.inf
    for fre in frequencys:
        tmp=glva.energy_predict_model.predict([[b,fre]])[0]
        if tmp<energy:
            energy=tmp
            f=fre                
    return b,f
def eais(model,tokenizer):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=1
    total_energy=0
    time_total=0
    num_step=0
    start_time=time.time()
    inital_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    while num_sum<glva.gl_max_count:
        tt1=time.time()
        batchsize,gpu_fre=eaisDecision()
        tt2=time.time()
        glva.result_txt.write("eais决策时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
        glva.result_txt.write("本次调节的batchsize: "+str(batchsize)+" frequency: "+str(gpu_fre)+"\n")
        waittime=1200*(time.time()-glva.gl_requestQ[0][1]+(batchsize-len(glva.gl_requestQ))/glva.sendspeed[0])
        glva.result_txt.write("预测最长等待时间(单位:ms): "+str(waittime)+"\n")
        glva.result_txt.write("当前请求速率："+str(glva.sendspeed[0])+"\n")
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
            tt2 = time.time()

            longest_wait_time=(tt2 - batch_times[0])*1000
            glva.result_txt.write("最长等待时间(单位:ms): "+str(longest_wait_time)+"\n")
            glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
            glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
            pynvml.nvmlDeviceSetGpuLockedClocks(glva.handle,135,gpu_fre)
            t1 = time.time()
            task=glva.pool.submit(do_predict,model,tokenizer,batch_inputs)
            while task.done()==False:
                power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                batch_energy+=power_gpu*2
                power_gpu=power_gpu/250
                time.sleep(0.001)
            slo, value1, value2 = task.result()
            t2 = time.time()
            num_step+=1
            glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str((t2 - t1) * 1000)+"\n")
            time_total += (t2 - t1) 
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
            glva.over_slo_[0]=num_overslo/(num_sum+0.0)
            i = i + len(batch_slo)        
    finish_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    end_time=time.time()-start_time
    base_energy=25*end_time*1000
    glva.result_txt.write("=====================推理总功耗："+str(finish_value-inital_value-base_energy)+" mJ \n")
def clipper(model,tokenizer):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=8
    total_energy=0
    time_total=0
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
            tt2 = time.time()
            glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
            glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
            t1 = time.time()
            task=glva.pool.submit(do_predict,model,tokenizer,batch_inputs)
            while task.done()==False:
                power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                batch_energy+=power_gpu*2
                power_gpu=power_gpu/250
                time.sleep(0.001)
            slo, value1, value2 = task.result()
            t2 = time.time()
            num_step+=1
            glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str((t2 - t1) * 1000)+"\n")
            time_total += (t2 - t1) 
            batch_energy = value2-value1
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
            glva.over_slo_[0]=num_overslo/(num_sum+0.0)
            i = i + len(batch_slo)
            if batch_slo[0]>glva.SLO:
                batchsize=batchsize-3
                batchsize= batchsize if batchsize>=1 else 1
            elif batch_slo[0]>0.9*glva.SLO:
                batchsize=int(batchsize*0.9)
                batchsize= batchsize if batchsize>=1 else 1
                continue
            else:
                batchsize+=1
    finish_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    end_time=time.time()-start_time
    base_energy=25*end_time*1000
    glva.result_txt.write("=====================推理总功耗："+str(finish_value-inital_value-base_energy)+" mJ \n")
def predict_rl(model,tokenizer,RL):
    num_overslo=0
    num_sum=0
    start=time.time()
    glva.result_txt.write("do_schedule start time: "+str(start)+"\n")
    i = 0
    batchsize=glva.batch_size[0]
    total_reward=0
    total_energy=0
    time_total=0
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
        batchsize=glva.action_GPU[action][0]
        gpu_fre = glva.action_GPU[action][1]
        done = 1
    
        glva.result_txt.write("**********调节频率的大小："+str(gpu_fre)+"\n")
        pynvml.nvmlDeviceSetGpuLockedClocks(glva.handle,135,gpu_fre)
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
            glva.result_txt.write("batch_times: "+str(batch_times)+"\n")
            longest_wait_time=(tt2 - batch_times[0])*1000
            glva.result_txt.write("最长等待时间(单位:ms): "+str(longest_wait_time)+"\n")
            
            glva.result_txt.write("数据读取及转换,GPU调频消耗时间(单位:ms): "+str((tt2 - tt1)*1000)+"\n")
            glva.result_txt.write("推理开始时间: "+str(time.time())+"\n")
            batch_energy=0
            sum_gpu_state=[0]
            rounds=0
            t1 = time.time()
            task=glva.pool.submit(do_predict,model,tokenizer,batch_inputs)
            while task.done()==False:
                power_gpu=pynvml.nvmlDeviceGetPowerUsage(glva.handle)/1000
                #对GPU功率进行归一化
                power_gpu=power_gpu/250
                gpu_state = [power_gpu]  # wym modify
                sum_gpu_state=[x+y for x, y in zip(sum_gpu_state,gpu_state)]
                rounds+=1
                time.sleep(0.002)
            slo, value1, value2 = task.result()
            t2 = time.time()
            num_step+=1
            batch_time=(t2 - tt1) * 1000
            glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str(batch_time)+"\n")
            
            time_total += (t2 - t1) 
            batch_energy = value2-value1
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
            glva.over_slo_[0]=num_overslo/(num_sum+0.0)
            i = i + len(batch_slo)
            # get next obserbation and reward
            time1=longest_wait_time
            longest_wait_time=1 if longest_wait_time>glva.SLO else longest_wait_time/glva.SLO
            batch_time=1 if batch_time>glva.SLO else batch_time/glva.SLO
            sum_gpu_state.append(round(longest_wait_time,3))
            sum_gpu_state.append(round(batch_time,3))
            currentspeed=glva.sendspeed[0]
            currentspeed=currentspeed/10
            sum_gpu_state.append(round(currentspeed,3))
            sum_gpu_state = [ round(i*10,3) for i in sum_gpu_state ]
            longest_wait_time=time1
            glva.result_txt.write("=====================当前状态："+str(sum_gpu_state)+"\n")
            obeservation_=sum_gpu_state  
            
            reward=0
            delay=batch_slo[0]
        
            energy=glva.energy_predict_model.predict([[batchsize,gpu_fre]])[0]
            glva.result_txt.write("=====================预测的单个推理功耗："+str(energy)+"\n")
            k=150
            
            if delay<=glva.SLO:
                reward=k*delay/energy
            else:
                reward=glva.SLO-delay
            obeservation=obeservation_
          
    finish_value=pynvml.nvmlDeviceGetTotalEnergyConsumption(glva.handle)
    end_time=time.time()-start_time
    base_energy=25*end_time*1000
    glva.result_txt.write("=====================推理总功耗："+str(finish_value-inital_value-base_energy)+" mJ \n")
def train(model,tokenizer,RL):
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
                batchsize=glva.action_GPU[action][0]
                gpu_fre = glva.action_GPU[action][1]
                max_q_value = RL.get_QValue(obeservation) * 0.005 + max_q_value * 0.995  # 平滑处理
                glva.result_txt.write("-------current step: "+str(num_step)+"\n")
                glva.result_txt.write("-------------Q Value: "+str(max_q_value)+"\n")
                rltime=time.time()-rltime
                while len(glva.gl_requestQ)<=batchsize:
                    continue
                if len(glva.gl_requestQ)>=batchsize:
                    finish=False
                    glva.result_txt.write("**********调节频率的大小："+str(gpu_fre)+"\n")
                    pynvml.nvmlDeviceSetGpuLockedClocks(glva.handle,135,gpu_fre)
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
                    batch_energy=glva.energy_predict_model.predict([[batchsize,gpu_fre]])[0]*batchsize
                    sum_gpu_state=[glva.power_predict_model.predict([[batchsize,gpu_fre]])[0]/250]
                    t2 = time.time()
                    num_step+=1
                    batch_time=glva.times_predict_model.predict([[batchsize,gpu_fre]])[0]*1000
                    glva.result_txt.write("--------------单批次时间消耗(单位:ms): "+str(batch_time)+"\n")
                    
                    time_total += batch_time
                    glva.result_txt.write("=====================单批次能耗(单位:mJ): "+str(batch_energy)+"\n")
                    total_energy+=batch_energy
                    glva.result_txt.write("=====================单个推理功耗："+str(batch_energy/batchsize)+"\n")

                    batch_slo = [(tt2-batch_times[n])*1000 + batch_time for n in range(batchsize)]
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
                    glva.over_slo_[0]=num_overslo/(num_sum+0.0)
                    i = i + len(batch_slo)
                    # get next obserbation and reward
                    
                    longest_wait_time=1 if longest_wait_time>glva.SLO else longest_wait_time/glva.SLO
                    batch_time=1 if batch_time>glva.SLO else batch_time/glva.SLO
                    sum_gpu_state.append(longest_wait_time)
                    sum_gpu_state.append(batch_time)
                    longest_wait_time=wait_time
                    currentspeed=glva.sendspeed[0]
                    currentspeed=currentspeed/10
                    sum_gpu_state.append(currentspeed)
                    sum_gpu_state = [ round(i*10,3) for i in sum_gpu_state ]
                    glva.result_txt.write("=====================当前状态："+str(sum_gpu_state)+"\n")
                    obeservation_=sum_gpu_state
                    reward=0
                    delay=batch_slo[0]
                    

                    energy=glva.energy_predict_model.predict([[batchsize,gpu_fre]])[0]
                    glva.result_txt.write("=====================预测的单个推理功耗："+str(energy)+"\n")

                    k=150
                    
                    if delay<=0.9*glva.SLO:
                        reward=k*delay/energy
                    else:
                        reward=(0.9*glva.SLO-delay)
                    reward = reward / 50
                
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
        self.model = ite
        self.tokenizer = datalen
        self.RL=RL
        self.RUN_MODE=glva.RUN_MODE
    def run(self):
        if self.RUN_MODE ==0:
            train(self.model,self.tokenizer,self.RL)
            torch.save(self.RL,glva.rlmodel)
        elif self.RUN_MODE==1:
            predict_rl(self.model,self.tokenizer,self.RL)
        elif self.RUN_MODE==2:
            clipper(self.model,self.tokenizer)
        elif self.RUN_MODE==3:
            eais(self.model,self.tokenizer)
        
        glva.result_txt.close()
        pynvml.nvmlShutdown()
        
        