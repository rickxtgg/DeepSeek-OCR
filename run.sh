#!/bin/bash
#该脚本运行方法：source run-dps.sh或者它的简写形式（一个点）. run-dps.sh
# 这个命令的作用是：让当前终端“读取”并执行run-dps.sh里的每一行命令，效果和你手动一行行输入完全一样。环境激活和目录切换都会保留。
# 如果使用bash方法就不行，因为bash会使用一个子shell运行，运行结束就自动没有了。
# 1. 设置网络
source /etc/network_turbo
source hunyuanocr/bin/activate
python 推理.py