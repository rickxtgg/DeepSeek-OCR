#!/bin/bash
#该脚本运行方法：source run-dps.sh或者它的简写形式（一个点）. run-dps.sh
# 这个命令的作用是：让当前终端“读取”并执行run-dps.sh里的每一行命令，效果和你手动一行行输入完全一样。环境激活和目录切换都会保留。
# 如果使用bash方法就不行，因为bash会使用一个子shell运行，运行结束就自动没有了。
# 1. 设置网络
source /etc/network_turbo
# 注意：如果您希望 vLLM 和 transformers 代码在同一环境下运行，则无需担心类似这样的安装错误：vllm 0.8.5+cu118 需要 transformers>=4.51.1
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
# 把conda虚拟环境安装在当前目录
conda create --prefix ./deepseek-ocr python=3.12.9 -y
# 激活环境
conda activate ./deepseek-ocr
cd DeepSeek-OCR
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu121-cp38-abi3-manylinux1_x86_64.whl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
# 有些环境缺失psutil库
pip install psutil
pip install flash-attn==2.7.3 --no-build-isolation