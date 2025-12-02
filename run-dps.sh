#!/bin/bash
#该脚本运行方法：source run-dps.sh或者它的简写形式（一个点）. run-dps.sh
# 这个命令的作用是：让当前终端“读取”并执行run-dps.sh里的每一行命令，效果和你手动一行行输入完全一样。环境激活和目录切换都会保留。
# 如果使用bash方法就不行，因为bash会使用一个子shell运行，运行结束就自动没有了。
# 1. 设置网络
source /etc/network_turbo

# 2. 初始化Conda到当前shell会话
# source /root/miniconda3/etc/profile.d/conda.sh

# 3. 激活自定义环境
# conda activate /root/autodl-tmp/deepseek-ocr
conda activate ./deepseek-ocr


# 4. 切换到工作目录
cd DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/

# 5. 这里开始写你需要在deepseek-ocr环境下运行的命令，例如：
# python app.py
# echo "环境已激活并切换到目标目录"