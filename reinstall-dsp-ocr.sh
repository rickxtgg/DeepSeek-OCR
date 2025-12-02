#!/bin/bash
#该脚本运行方法：source run-dps.sh或者它的简写形式（一个点）. run-dps.sh
# 这个命令的作用是：让当前终端“读取”并执行run-dps.sh里的每一行命令，效果和你手动一行行输入完全一样。环境激活和目录切换都会保留。
# 如果使用bash方法就不行，因为bash会使用一个子shell运行，运行结束就自动没有了。
# 1. 设置网络
source /etc/network_turbo
# 注意：如果您希望 vLLM 和 transformers 代码在同一环境下运行，则无需担心类似这样的安装错误：vllm 0.8.5+cu118 需要 transformers>=4.51.1
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
# 复制模型
cp -r deepseek-ai /root/autodl-tmp/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/
# 复制要处理的pdf目录
cp -r input_pdf_batch /root/autodl-tmp/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/
# 复制修改脚本
cp -r auto_fix_t4_compatibility.py /root/autodl-tmp/DeepSeek-OCR/
# 复制批量处理pdf脚本
cp -r run_dpsk_ocr_pdf_batch.py /root/autodl-tmp/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/
# 把安装在当前目录的conda虚拟环境激活
conda activate ./deepseek-ocr
cd DeepSeek-OCR
