# encoding: utf-8

import sys
import argparse
import logging
import os
import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm
tqdm.monitor_interval = 0

import torch.backends.cudnn as cudnn
import collections

from utils import utils
from utils.utils import str2bool
import data_loader
from model import HCN

from defense import *
import setGPU

from feeder.feeder import StoreDataFeeder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data/zth/ntu', help="root directory for all the datasets")
parser.add_argument('--dataset_name', default='NTU-RGB-D-CV', help="dataset name ") # NTU-RGB-D-CS,NTU-RGB-D-CV
parser.add_argument('--model_dir', default='./',
                    help="parents directory of model")

parser.add_argument('--model_name', default='HCN',help="model name")
parser.add_argument('--load_model',
        help='Optional, load trained models')
parser.add_argument('--load',
        type=str2bool,
        default=False,
        help='load a trained model or not ')
parser.add_argument('--mode', default='train', help='train,test,or load_train')
parser.add_argument('--num', default='01', help='num of trials (type: list)')

parser.add_argument('--sigma', type=float, default=0.1, help='std of noise for defense')

parser.add_argument('--targeted', type=str2bool, default=False, help='targeted or untargeted')
parser.add_argument('--target_label', type=int, default=0, help='target label')
parser.add_argument('--beta', type=float, default=1.0, help='beta')

parser.add_argument('--apply_defense', type=str2bool, default=False, help='beta')
parser.add_argument('--filter_num', type=int, default=5, help='1x5 or 1x7')


def evaluate(model, loss_fn, dataloader, metrics, params, sigma=0.01, apply_defense=False):

    # set model to evaluation mode
    model.eval()
    Rdefense = NTU_DEFENSE(sigma=sigma, num_samples=50)

    # summary for current eval loop
    summ = []

    ## to save certified radii
    certified_radii = []


    # compute metrics over the dataset
    for data, label in dataloader:

        # move to GPU if available
        if params.cuda:
            if params.data_parallel:
                data, label = data.cuda(), label.long().cuda()
            else:
                data, label = data.cuda(params.gpu_id), label.long().cuda(params.gpu_id)


        ## create primal variables and lagrangian multipliers
        data = Variable(data, requires_grad=False)
        label = Variable(label, requires_grad=False)

        # radii = Rdefense.certification(model, data, label)
        if apply_defense:
            output = Rdefense.randomized_smoothing(model, data)
        else:
            output = model(data)

        # store all metrics on this batch
        summary_batch = {metric: metrics[metric](output, label)
                         for metric in metrics}

        summ.append(summary_batch)
        print(summary_batch, flush=True)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}

    print(metrics_mean)



if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    experiment_path =  os.path.join(args.model_dir,'experiments',args.dataset_name,args.model_name+args.num)
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)


    json_file = os.path.join(experiment_path,'params.json')
    if not os.path.isfile(json_file):
        with open(json_file,'w') as f:
            print("No json configuration file found at {}".format(json_file))
            f.close()
            print('successfully made file: {}'.format(json_file))

    params = utils.Params(json_file)

    if args.load :
        print("args.load=",args.load)
        if args.load_model:
            params.restore_file = args.load_model
        else:
            params.restore_file = experiment_path + '/checkpoint/best.pth.tar'


    params.dataset_dir = args.dataset_dir
    params.dataset_name = args.dataset_name
    params.model_version = args.model_name
    params.experiment_path = experiment_path
    params.mode = args.mode
    if params.gpu_id >= -1:
        params.cuda = True

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    if params.gpu_id >= -1:
        torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = False
    cudnn.benchmark = True # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.cuda.empty_cache() # release cache

    port,env = 8097,params.model_version
    columnnames,rownames = list(range(1,params.model_args["num_class"]+1)),list(range(1,params.model_args["num_class"]+1))



    ## load data
    if args.targeted:
        data_path = params.experiment_path+'/beta_' + str(int(args.beta)) + '_target_'+str(args.target_label)+'_adv_data.npz'
    else:
        data_path = params.experiment_path+'/beta_' + str(int(args.beta)) + '_untargeted_adv_data.npz'


    test_dl = torch.utils.data.DataLoader(
        dataset=StoreDataFeeder(data_path),
        batch_size=params.batch_size ,
        shuffle=False,
        num_workers=params.num_workers,pin_memory=params.cuda)


    if 'HCN' in params.model_version:
        model = HCN.HCN(**params.model_args)
        if params.data_parallel:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(params.gpu_id)

        loss_fn = HCN.loss_fn
        metrics = HCN.metrics


    ### load model
    restore_file = params.restore_file
    if restore_file is not None:
        checkpoint = utils.load_checkpoint(restore_file, model)

        best_val_acc = checkpoint['best_val_acc']
        params.current_epoch = checkpoint['epoch']
        print('best_val_acc=',best_val_acc)
    else:
        raise


    evaluate(model, loss_fn, test_dl, metrics, params, sigma=args.sigma, apply_defense=args.apply_defense)
