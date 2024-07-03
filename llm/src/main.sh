#!/bin/bash
#!/usr/bin/python

export CUDA_VISIBLE_DEVICES=0

PM="/home/PerformanceMeasurement/PerfMeasure.bin"
# # source "/home/lx/anaconda3/etc/profile.d/conda.sh"
# /home/lx/anaconda3/bin/activate /home/lx/anaconda3/envs/tensorrt
SCRIPTS_LIST=(
    
    "start.py"

)

SLO=10000
sends_per_second=270
# sends_speed_file="wiki.txt"
sends_speed_file="dynamic.txt"
# sends_speed_file="sogouQ.txt"
# sends_speed_file="speeds.txt"
# sends_speed_file="trainspeed.txt"
# sends_speesd_file="sogouQ.txt"
RUN_MODE=2
#!/bin/bash

# 定义一个变量来存储第一行内容
# first_line=""

# 使用while循环只读取文件的第一行
# while IFS= read -r line && [ -n "$line" ]; do
#   first_line="$line"
#   break # 读取完第一行后退出循环
# done < $sends_speed_file
# # 现在变量first_line中已经存储了txt文件的第一行内容
# echo "The first line is: $first_line"

# save_path="./qwen-int4/eaisdynamic"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="densenet/train.pth"
# PM_OUT="./qwen-int4/eaisdynamicstatus.txt"

# save_path="./qwen-int4/speeds"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./qwen-int4/speedsstatus.txt"

# save_path="./qwen-int4/train"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"
# PM_OUT="./qwen-int4/status.txt"

save_path="./qwen-int4/clipperdynamic"
trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
rlmodel="qwen-int4/train.pth"
PM_OUT="./qwen-int4/clipperdynamicstatus.txt"


# save_path="./qwen-int4/dynamic"
# trtFile="/home/model/qwen/Qwen-7B-Chat-Int4"
# rlmodel="qwen-int4/train.pth"
# PM_OUT="./qwen-int4/dynamicstatus.txt"



# GPU index, 功率/频率配置, 运行次数
GPUName="V100S"
GPUIndex="0"
TuneType="SM_RANGE"

SampleInterval="100"
PowerThreshold="25"
# PMFlagBase="-e -i "${GPUIndex}" -s "${SampleInterval}" -t "${PowerThreshold}" -tune SM_RANGE "
PMFlagBase="-e -i "${GPUIndex}" -s "${SampleInterval}" -t "${PowerThreshold}" -m DAEMON -trace"
# -e -i 1 -s 100 -t 1.65 -m DAEMON

SleepInterval="5"

source "/home/PerformanceMeasurement/Msg2EPRT.sh"


# 启动运行时
echo "ExitMeasurement"
ExitMeasurement

echo "sleep 1s"
sleep 1s

echo "sudo ${PM} ${PMFlagBase} &"
${PM} ${PMFlagBase} &

echo "sleep 1s"
sleep 1s


echo "ResetMeasurement ${PM_OUT}"
ResetMeasurement ${PM_OUT}
sleep 1s
echo "StartMeasurement"
StartMeasurement

echo "sleep 1s"
sleep 1s

# # execute scripts
for Script in ${SCRIPTS_LIST[@]};
do
    python ${Script} "--SLO" ${SLO} "--sends_per_second" ${sends_per_second} \
    "--sends_speed_file" ${sends_speed_file} "--RUN_MODE" ${RUN_MODE} \
    "--save_path" ${save_path} "--trtFile" ${trtFile} "--rlmodel" ${rlmodel}
done


echo "StopMeasurement"
StopMeasurement

echo "sleep 1s"
sleep 1s

echo "ExitMeasurement"
ExitMeasurement

echo "sleep 1s"
sleep 1s
echo "测试完毕"