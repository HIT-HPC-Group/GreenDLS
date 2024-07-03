#!/bin/bash
#!/usr/bin/python

export CUDA_VISIBLE_DEVICES=0

PM="/home/PerformanceMeasurement/PerfMeasure.bin"
# # source "/home/lx/anaconda3/etc/profile.d/conda.sh"
# /home/lx/anaconda3/bin/activate /home/lx/anaconda3/envs/tensorrt
SCRIPTS_LIST=(
    
    # "/home/wfr/work/Energy/EPOpt/Bench/CANDLE.sh"
    "start.py"

)

SLO=200
sends_per_second=270
sends_speed_file="wiki.txt"
# sends_speed_file="dynamic.txt"
# sends_speed_file="sogouQ.txt"
# sends_speed_file="speeds.txt"
# sends_speed_file="trainspeed.txt"
RUN_MODE=1
# 0:TRAIN 1:TEST 2:Clipper 3:EAIS
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


# save_path="./resnet50/eaisdynamic"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/eaisdynamicstatus.txt","w"

# save_path="./resnet50/dynamic"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/dynamicstatus.txt"

# save_path="./resnet50/clipperdynamic"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/clipperdynamicstatus.txt"

# save_path="./resnet50/clipperwiki"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/clipperwikistatus.txt"

# save_path="./resnet50/clippersougouQ"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/clippersougouQstatus.txt"

# save_path="./resnet50/sougouQ"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/sougouQstatus.txt"

# save_path="./resnet50/eaissougouQ"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/eaissougouQstatus.txt"

# save_path="./resnet50/eaiswiki"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/eaiswikistatus.txt"

save_path="./resnet50/wiki"
trtFile="/home/model/resnet50.plan"
rlmodel="resnet50/train.pth"
PM_OUT="./resnet50/wikistatus.txt"

# save_path="./resnet50/train"
# trtFile="/home/model/resnet50.plan"
# rlmodel="resnet50/train.pth"
# PM_OUT="./resnet50/status.txt"


# save_path="./vgg19/eaisdynamic"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="./vgg19/eaisdynamicstatus.txt"

# save_path="./vgg19/clipperdynamic"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="./vgg19/clipperdynamicstatus.txt"

# save_path="./vgg19/eaissougouQ"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/eaissougouQstatus.txt"

# save_path="./vgg19/clippersougouQ"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/clippersougouQstatus.txt"

# save_path="./vgg19/sougouQ"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/sougouQstatus.txt"

# save_path="./vgg19/eaiswiki"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/eaiswikistatus.txt"

# save_path="./vgg19/clipperwiki"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/clipperwikistatus.txt"

# save_path="./vgg19/wiki"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/wikistatus.txt"

# save_path="./vgg19/dynamic"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="vgg19/dynamicstatus.txt"

# save_path="./vgg19/train"
# trtFile="/home/model/vgg19.plan"
# rlmodel="vgg19/train.pth"
# PM_OUT="./vgg19/status.txt"

# save_path="./densenet/eaiswiki"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/eaiswikistatus.txt"

# save_path="./densenet/eaisdynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/eaisdynamicstatus.txt"

# save_path="./densenet/speeds"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/speedsstatus.txt"

# save_path="./densenet/clipperwiki"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/clipperwikistatus.txt"

# save_path="./densenet/wiki"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/wikistatus.txt"

# save_path="./densenet/clippersougouQ"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/clippersougouQstatus.txt"

# save_path="./densenet/eaissougouQ"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/eaissougouQstatus.txt"

# save_path="./densenet/clipperdynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/clipperdynamicstatus.txt"

# save_path="./densenet/eaisdynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/eaisdynamicstatus.txt"

# save_path="./densenet/dynamic"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/dynamicstatus.txt"

# save_path="./densenet/sougouQ"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/sougouQstatus.txt"

# save_path="./densenet/train"
# trtFile="/home/model/densenet201.plan"
# rlmodel="densenet/train.pth"
# PM_OUT="./densenet/status.txt"

# GPU index, 功率/频率配置, 运行次数
GPUName="3080Ti"
GPUIndex="0"
TuneType="SM_RANGE"

SampleInterval="100"
PowerThreshold="29"
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