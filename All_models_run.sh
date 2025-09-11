#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/chenruixi/miniconda3/envs/BGPrediction/bin/python
ROOT=/home/chenruixi/BGPredition

# 设置软链接，确保 ../Glucose_Data 指向正确目录（只需做一次，已存在则跳过）
if [ ! -L /home/chenruixi/Glucose_Data ]; then
    ln -s /home/chenruixi/BGPredition/Glucose_Data /home/chenruixi/Glucose_Data
fi

# 需要训练的模型列表
models=(
    Crossformer
    PatchTST
    TimeXer
    DLinear
    Informer
    Transformer
    LSTM
    GRU
    Glucoformer
)

for model in "${models[@]}"
do
    echo "---------------- Running $model ----------------"
    cd $ROOT/$model         # 进入模型自己的目录
    $PYTHON ${model}_main.py
    if [ $? -ne 0 ]; then
        echo "⚠️ $model 运行失败"
        exit 1
    fi
    cd $ROOT                # 回到项目根目录，准备下一个模型
done