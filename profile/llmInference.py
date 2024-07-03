from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
from modelscope.msdatasets import MsDataset
import pynvml
import sys
import torch
import logging
import time
# model_path = "/home/model/qwen/Qwen-7B-Chat"
model_path = "/home/model/qwen/Qwen-7B-Chat-Int4"
sys.path.append(model_path)
# model_path = ""
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
data_path = '/home/data/iic/IWSLT-Chinese-to-English-Machine-Translation-Spoken/master/data_files/extracted'
batchsizeList=[1,2,4,8,16,32,64]
frequencylist=(1597,1477,1357,1237,1117,997,877,757,637,517)
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

def load_data(data_path):
    ds =  MsDataset.load(data_path, subset_name='default', split='train')
    return ds

def get_batch_data(ds,batchsize):
    rawBatchInputList = []
    tmpDsList = []
    for item in ds:
        # if batchsize==2:
        #     print(item)
        #     print(type(item))
        tmpDsList.append("翻译下列句子到英文 :"+item['0'])
        tmpDsList.append("翻译下列句子到中文 :"+item['1'])
    length=len(tmpDsList)
    for i in range(length):
        if i+batchsize>=length:
            break
        tmpList=[]
        for j in range(batchsize):
            tmpList.append(tmpDsList[i+j])
        rawBatchInputList.append(tmpList)
    return rawBatchInputList

def predict(tokenizer,model,all_raw_text):
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
    # padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]

    # batch_response = [
    #     decode_tokens(
    #         batch_out_ids[i][padding_lens[i]:],
    #         tokenizer,
    #         raw_text_len=len(batch_raw_text[i]),
    #         context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
    #         chat_format="chatml",
    #         verbose=False,
    #         errors='replace'
    #     ) for i in range(len(all_raw_text))
    # ]

if __name__=="__main__":
    # logging.basicConfig(filename='qwen7B-INT4_result.txt',filemode='w',level=logging.INFO)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    model,tokenizer = load_model(model_path)
    ds = load_data(data_path)
    predict(tokenizer,model,["预热模型"])
    for frequency in frequencylist:
        pynvml.nvmlDeviceSetGpuLockedClocks(handle,frequency,frequency)
        for batchsize in batchsizeList:
            # logging.info("frequency:%d,batch_size:%d\n"%(frequency,batchsize))
            batchDsList = get_batch_data(ds,batchsize)
            batchDsList = batchDsList[0:20]
            n=5
            energy=0
            exectime=0
            for i in range(n):
                start = time.time()
                value1 = pynvml.nvmlDeviceGetPowerUsage(handle)
                for d in batchDsList:
                    predict(tokenizer,model,d)
                end = time.time()
                value2 = pynvml.nvmlDeviceGetPowerUsage(handle)
                energy+=(value2-value1)/(len(batchDsList)*batchsize)
                exectime+=(end-start)/len(batchDsList)
                time.sleep(1)
            # logging.info("avgtime: %.4fs\n"%(exectime/n))
            # logging.info("avgenery: %.4fmJ\n"%(energy/n))
    
    