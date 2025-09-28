import torch
import time
import logging
from collections import OrderedDict
import re
import matplotlib
from matplotlib import pyplot as plt
from os.path import split, splitext

def get_logger(filename, mode='a'):
    # 创建日志记录器
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger

class RunningAverage(object):

    def __init__(self, *keys):
        self.sum = OrderedDict() # 创建有序字典来保存各指标的累积和
        self.cnt = OrderedDict() # 创建有序字典来记录各指标的计数（即更新次数）
        self.clock = time.time() # 记录初始化时间，用于计算运行时间
        # 初始化
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        # 转换格式
        if isinstance(val, torch.Tensor):
            val = val.item()
        # 更新指标
        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        # 重置所有指标的累积和和计数
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0
        # 重置计时器
        self.clock = time.time()

    def clear(self):
        # 清除全部
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()

    def keys(self):
        return self.sum.keys() # 返回所有指标的键

    def get(self, key):
        assert(self.sum.get(key, None) is not None) # 请求的键必须存在
        return self.sum[key] / self.cnt[key] # 返回指定指标的平均值

    def info(self, show_et=True):
        # 记录精度
        line = ''
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]
            if type(val) == float:
                line += f'{key} {val:.4f} '
            else:
                line += f'{key} {val} '.format(key, val)
        if show_et:
            line += f'({time.time()-self.clock:.3f} secs)'
        return line

def get_log(fileroot):
    # 从日志文件中读取性能指标和时间信息
    step = []
    loss = []
    train_time = []
    eval_time = []
    ctxll = []
    tarll = []
    file = open(fileroot, "r")
    lines = file.readlines()
    for line in lines:
        # training step
        if "step" in line:
            linesplit = line.split(" ")
            step += [int(linesplit[3])]
            _loss = linesplit[-3]
            loss += [100 if _loss=="nan" else float(_loss)]
            train_time += [float(linesplit[-2][1:])]
        # evaluation step
        elif "ctx_ll" in line:
            linesplit = line.split(" ")
            ctxll += [float(linesplit[-5])]
            tarll += [float(linesplit[-3])]
            eval_time += [float(linesplit[-2][1:])]
    
    return step, loss, None, ctxll, tarll


def plot_log(fileroot, x_begin=None, x_end=None):
    step, loss, stepll, ctxll, tarll = get_log(fileroot)
    step = list(map(int, step))
    loss = list(map(float, loss))
    ctxll = list(map(float, ctxll))
    tarll = list(map(float, tarll))
    stepll = list(map(int, stepll)) if stepll else None
    
    if x_begin is None:
        x_begin = 0
    if x_end is None:
        x_end = step[-1]
    
    print_freq = 1 if len(step)==1 else step[1] - step[0]

    plt.clf()
    plt.plot(step[x_begin//print_freq:x_end//print_freq],
             loss[x_begin//print_freq:x_end//print_freq])
    plt.xlabel('step')
    plt.ylabel('loss')

    dir, file = split(fileroot)
    filename = splitext(file)[0]
    plt.savefig(dir + "/" + filename + f"-{x_begin}-{x_end}.png")
    plt.clf()  # clear current figure