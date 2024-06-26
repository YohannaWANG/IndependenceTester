#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 19 15:33:02 2023

@author: yohanna
"""

'''
data arguments
n: number of nodes
s: number of samples
d: average degree of node
'''
n = 30
s = '100'
d = 1
batch = 20

'''
data arguments
choice: choice which real bnlearn data to load. 
load: choice whether load synthetic data or real bnlearn data
options: 'syn', 'real'
options: 'ecoli70', 'magic-niab', 'magic-irri', 'arth150', #{'healthcare', 'sangiovese', 'mehra'}
'''
load = 'syn'
choice = 'arth150'

'''
data arguments
tg: type of graph, options: chain, er, sf, rt
tn: type of noise, options: ev, uv, ca, ill, exp, gum
'''
tg = 'rt'
tn = 'ev'

'''
data arguments
sf: scaling factor for range of binary DAG adjacency
'''
sf = 1.0

'''
miscellaneous arguments
seed: seed value for randomness
log: log on file (True) or print on console (False)
gpu: gpu number, options: 0, 1, 2, 3, 4, 5, 6, 7
'''
seed = 5
log = False
gpu = 0

import argparse


def parse():
    '''add and parse arguments / hyperparameters'''
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Chain Graph Structure Learning from Observational Data")

    p.add_argument('--n', type=int, default=n, help='number of nodes')
    #p.add_argument('--s', type=int, default=s, help='number of samples')
    p.add_argument('--s', nargs='+', help="a list of sample numbers")
    p.add_argument('--d', type=int, default=d, help='average degree of node')
    p.add_argument('--batch', type=int, default=batch, help='number of batch size')

    p.add_argument('--choice', type=str, default=choice, help='choose which real bnlearn data to load')
    p.add_argument('--load', type=str, default=load, help='either load synthetic or real data')

    p.add_argument('--tg', type=str, default=tg, help='type of graph, options: rt')
    p.add_argument('--tn', type=str, default=tn, help='type of noise, options: ev, uv, exp, gum')

    p.add_argument('--sf', type=float, default=sf, help='scaling factor for range of binary DAG adjacency')

    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return True

    p.add_argument('--seed', type=int, default=seed, help='seed value for randomness')
    p.add_argument("--log", type=str2bool, default=log, help="log on file (True) or print on console (False)")
    p.add_argument('--gpu', type=int, default=gpu, help='gpu number, options: 0, 1, 2, 3, 4, 5, 6, 7')

    p.add_argument('-f')  # for jupyter default
    return p.parse_args()


import os, inspect, logging, uuid


class Logger():
    def __init__(self, p):
        '''Initialise logger '''

        # setup log checkpoint directory
        current = os.path.abspath(inspect.getfile(inspect.currentframe()))
        Dir = os.path.join(os.path.split(os.path.split(current)[0])[0], "checkpoints")
        self.log = p.log

        # setup log file
        if self.log:
            if not os.path.exists(Dir): os.makedirs(Dir)
            name = str(uuid.uuid4())

            Dir = os.path.join(Dir, name)
            if not os.path.exists(Dir): os.makedirs(Dir)
            p.dir = Dir

            # setup logging
            logger = logging.getLogger(__name__)

            file = os.path.join(Dir, name + ".log")
            logging.basicConfig(format="%(asctime)s - %(levelname)s -   %(message)s", filename=file, level=logging.INFO)
            self.logger = logger

    # function to log
    def info(self, s):
        if self.log:
            self.logger.info(s)
        else:
            print(s)


import torch, numpy as np, random


def setup():
    # parse arguments
    p = parse()
    p.logger = Logger(p)
    D = vars(p)

    # log configuration arguments
    l = [''] * (len(D) - 1) + ['\n\n']
    p.logger.info("Arguments are as follows.")
    for i, k in enumerate(D): p.logger.info(k + " = " + str(D[k]) + l[i])

    # set seed
    s = p.seed
    print('s', s)
    random.seed(s)
    print('random.seed', random.seed(s))
    # torch.manual_seed(s)
    # p.gen = torch.Generator().manual_seed(s)

    np.random.seed(s)
    print('np.random.seed(s)', np.random.seed(s))
    p.rs = np.random.RandomState(s)
    print('p.rs', p.rs)
    os.environ['PYTHONHASHSEED'] = str(s)

    # set device (gpu/cpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.gpu)
    p.device = torch.device('cuda') if p.gpu != '-1' and torch.cuda.is_available() else torch.device('cpu')

    return p


if __name__ == '__main__':
    setup()