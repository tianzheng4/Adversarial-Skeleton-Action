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

parser.add_argument('--sigma', type=float, default=0.1, help='std of noise for certification')

## data_batch: [64, 3, 32, 25, 2], labels_batch: [64]

def certify(model, loss_fn, dataloader, metrics, params, logger, sigma=0.1, outfile=None):

    # set model to evaluation mode
    model.eval()
    Rdefense = NTU_DEFENSE(sigma=sigma, num_samples=1000)

    # summary for current eval loop
    summ = []

    ## to save certified radii
    certified_radii = []

    f = open(outfile, 'w')
    print("radius", file=f, flush=True)

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

        radii = Rdefense.certification(model, data, label)


        # store all metrics on this batch
        summary_batch = {}

        summary_batch['r01'] = np.sum(np.greater(radii, 0.1).astype(int))
        summary_batch['r02'] = np.sum(np.greater(radii, 0.2).astype(int))
        summary_batch['r03'] = np.sum(np.greater(radii, 0.3).astype(int))

        summary_batch['r05'] = np.sum(np.greater(radii, 0.5).astype(int))
        summary_batch['r10'] = np.sum(np.greater(radii, 1.0).astype(int))

        summ.append(summary_batch)
        print(summary_batch, flush=True)

        ## record all certified radii

        for i in range(radii.shape[0]):
            print("{}".format(radii[i]), file=f, flush=True)


    # compute mean of all metrics in summary
    metrics_sum = {metric: np.sum([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_sum.items())
    logger.info("- Eval metrics : " + metrics_string)

    f.close()


    return metrics_sum





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
    torch.backends.cudnn.deterministic = False # must be True to if you want reproducible,but will slow the speed

    cudnn.benchmark = True # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.cuda.empty_cache() # release cache
    # Set the logger
    utils.set_logger(os.path.join(experiment_path, 'certify_sigma_'+ str(args.sigma).replace('.','')+'.log'))

    logger = logging.getLogger()

    port,env = 8097,params.model_version
    columnnames,rownames = list(range(1,params.model_args["num_class"]+1)),list(range(1,params.model_args["num_class"]+1))


    # log all params
    d_args = vars(args)
    for k in d_args.keys():
        logging.info('{0}: {1}'.format(k, d_args[k]))
    d_params = vars(params)
    for k in d_params.keys():
        logger.info('{0}: {1}'.format(k, d_params[k]))


    if 'HCN' in params.model_version:
        model = HCN.HCN(**params.model_args)
        if params.data_parallel:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(params.gpu_id)

        loss_fn = HCN.loss_fn
        metrics = HCN.metrics


    logger.info(model)
    # Create the input data pipeline
    logger.info("Loading the datasets...")
    # fetch testing dataloaders
    test_dl = data_loader.fetch_dataloader('test', params)
    logger.info("- done.")

    ### load model
    restore_file = params.restore_file
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        checkpoint = utils.load_checkpoint(restore_file, model)

        best_val_acc = checkpoint['best_val_acc']
        params.current_epoch = checkpoint['epoch']
        print('best_val_acc=',best_val_acc)
    else:
        raise

    outfile = os.path.join(params.experiment_path, 'sigma_'+str(args.sigma))
    certify(model, loss_fn, test_dl, metrics, params, logger, sigma=args.sigma, outfile=outfile)
