ADMMattack# encoding: utf-8

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

from attack import *
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



parser.add_argument('--targeted', type=str2bool, default=False, help='targeted or untargeted')
parser.add_argument('--target_label', type=int, default=0, help='target label')
parser.add_argument('--num_steps', type=int, default=100, help='num of steps')
parser.add_argument('--beta', type=float, default=1.0, help='beta')



## data_batch: [64, 3, 32, 25, 2], labels_batch: [64]

def attack(model, loss_fn, dataloader, metrics, params, logger,
                            num_steps=100, target_label=None, targeted=False,
                            beta=1.0, add_noise=False, noise_sigma=0.01,
                            apply_defense=False):

    # set model to evaluation mode
    model.eval()
    if 'NTU' in params.dataset_name:
        ADMMattack = NTU_ADMM_ATTACK(num_frames=params.test_feeder_args['window_size'])
    elif 'Kinetics' in params.dataset_name:
        ADMMattack = Kinetics_ADMM_ATTACK(num_frames=params.test_feeder_args['window_size'])
    else:
        raise
    # Rdefense = NTU_DEFENSE(sigma=0.1)

    # summary for current eval loop
    summ = []

    ## to save adversarial and original skeletons
    adv_skeletons = []
    original_skeletons = []
    skeletons_indice = []
    original_labels = []

    # compute metrics over the dataset
    for data, label in dataloader:

        ## store original data and label
        original_skeletons.append(data.data.numpy())
        original_labels.append(label.data.numpy())

        bone_length, joint_bone1, joint_bone2, velocity = ADMMattack.preprocess(data)

        lambda_bone = torch.zeros_like(bone_length)
        lambda_joint = torch.zeros_like(joint_bone1)
        lambda_velocity = torch.zeros(velocity.size(0), velocity.size(2),
                                      velocity.size(3), velocity.size(4))

        if targeted:
            if target_label is None:
                raise ValueError('pls set target label')
            else:
                label = torch.ones(*label.size())*target_label


        data_adv = data.clone()
        # move to GPU if available
        if params.cuda:
            if params.data_parallel:
                data, data_adv, label = data.cuda(), data_adv.cuda(), label.long().cuda()
                lambda_bone, lambda_joint, lambda_velocity = lambda_bone.cuda(), lambda_joint.cuda(), lambda_velocity.cuda()
            else:
                data, data_adv, label = data.cuda(params.gpu_id), data_adv.cuda(params.gpu_id), label.long().cuda(params.gpu_id)


        bone_length, joint_bone1, joint_bone2, velocity = ADMMattack.preprocess(data.data)

        ## since there might not be the second person in a frame, we need to initialize a mask
        ## bone_length [64, 32, 24, 2]
        mask = bone_length.abs().max(2)[0]
        mask = torch.sign(torch.max(mask - 1e-5, torch.zeros_like(mask))).float()
        mask = mask.view(data.size(0), 1, data.size(2), 1, data.size(4))
        mask = mask.repeat(1, data.size(1), 1, data.size(3), 1).cuda()
        sum_unmasked_joints = torch.sum(mask)

        ## create primal variables and lagrangian multipliers
        data = Variable(data, requires_grad=False)
        data_adv = Variable(data_adv, requires_grad=True)
        label = Variable(label, requires_grad=False)

        lambda_bone = Variable(lambda_bone, requires_grad=True)
        lambda_joint = Variable(lambda_joint, requires_grad=True)
        lambda_velocity = Variable(lambda_velocity, requires_grad=True)

        ## inner minimization
        optimizer_data = optim.Adam([data_adv], lr=0.001)

        if add_noise:
            data_adv.data += torch.randn_like(data_adv.data)*noise_sigma*mask

        for step in range(num_steps):

            for sub_step in range(10):
                output = model(data_adv)

                loss_ent = loss_fn(output, label, params=params)['ls_CE']
                ## argumented lagrangian
                loss_bone = ADMMattack.bone_constraints(data_adv, bone_length, mask=mask)
                loss_bone = torch.mean(lambda_bone*loss_bone) + 0.5*beta*torch.mean(loss_bone*loss_bone)

                loss_joint = ADMMattack.angle_constraints(data, data_adv, joint_bone1, joint_bone2, 0.1, mask=mask)
                loss_joint = torch.mean(lambda_joint*loss_joint) + 0.5*beta*torch.mean(loss_joint*loss_joint)

                loss_smooth = ADMMattack.temporal_smoothness_constraints(data_adv, velocity, epsilon=0.1, mask=mask)
                loss_smooth = torch.mean(lambda_velocity*loss_smooth) + 0.5*beta*torch.mean(loss_smooth*loss_smooth)


                if targeted:
                    loss = loss_ent + loss_bone + loss_joint + loss_smooth
                else:
                    if 'NTU' in params.dataset_name:
                        loss = logit_loss(output, label, num_classes=60) + loss_bone + loss_joint + loss_smooth
                    elif 'Kinetics' in params.dataset_name:
                        loss = logit_loss(output, label, num_classes=400) + loss_bone + loss_joint + loss_smooth

                optimizer_data.zero_grad()
                loss.backward()
                optimizer_data.step()

                data_adv.data = data_adv.data*mask

            ## update multipliers
            lambda_bone.data += beta*ADMMattack.bone_constraints(data_adv, bone_length, mask=mask).data
            lambda_joint.data += beta*ADMMattack.angle_constraints(data,
                                                data_adv, joint_bone1, joint_bone2, 0.1, mask=mask).data
            lambda_velocity.data += beta*ADMMattack.temporal_smoothness_constraints(
                                                 data_adv, velocity, epsilon=0.1, mask=mask).data

        ## Compute the output (logit)
        if not apply_defense:
            output = model(data_adv)
        else:
            output = Rdefense.randomized_smoothing(model, data_adv, mask=mask)

        ## compute metrics
        loss = loss_fn(output, label, params=params)['ls_CE']
        bone_vio_rate = ADMMattack.bone_vio_rate(data_adv, bone_length, mask=mask)
        joint_vio_rate = ADMMattack.angle_vio_rate(data, data_adv, joint_bone1, joint_bone2, mask=mask)
        smooth_vio_rate = torch.sum(ADMMattack.temporal_smoothness_constraints(
                                        data_adv, velocity, epsilon=0.1, mask=mask))/sum_unmasked_joints
        kinetics_vio_rate = ADMMattack.kinetics_vio_rate(data_adv, velocity)


        # store all metrics on this batch
        summary_batch = {metric: metrics[metric](output, label)
                         for metric in metrics}

        summary_batch['bone_vio_rate'] = bone_vio_rate.data.item()
        summary_batch['joint_vio_rate'] = joint_vio_rate.data.item()
        summary_batch['smooth_vio_rate'] = smooth_vio_rate.data.item()
        summary_batch['kinetics_rate'] = kinetics_vio_rate.data.item()
        summary_batch['L2_distance'] = ADMMattack.L2_distance(data, data_adv).data.item()

        ## record some scalars
        adv_skeletons.append(data_adv.data.cpu().numpy())

        summ.append(summary_batch)

    original_skeletons = np.concatenate(original_skeletons)
    adv_skeletons = np.concatenate(adv_skeletons)
    original_labels = np.concatenate(original_labels)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print(metrics_mean)
    metrics_string = " ; ".join("{}: {}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Eval metrics : " + metrics_string)



    if targeted:
        np.savez(params.experiment_path+'/beta_' + str(int(beta)) + str(int(beta*10)) + '_target_'+str(target_label)+'_adv_data.npz', natdata=original_skeletons,
                                            advdata=adv_skeletons, natlabels=original_labels)
    else:
        np.savez(params.experiment_path+'/beta_' + str(int(beta)) + str(int(beta*10)) +'_untargeted_adv_data.npz', natdata=original_skeletons,
                                            advdata=adv_skeletons, natlabels=original_labels)


    return metrics_mean

def logit_loss(output, label, targeted=False, device=None, num_classes=60):
    # compute the probability of the label class versus the maximum other
    label_cpu = label.cpu().data
    onehot = torch.zeros(label_cpu.size(0), num_classes).scatter_(1, label_cpu.unsqueeze(1), 1.)
    if not device:
        onehot = onehot.float().cuda()
    real = (onehot * output).sum(1)
    other = ((1. - onehot) * output - onehot * 10000.).max(1)[0]

    confidence = 1.0
    if targeted:
        loss = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    else:
        ## if non-targeted other - real = confidence > 0 (when loss = 0)
        ## loss is larger than 0 because real is larger than other
        loss = torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)

    return loss.mean()





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
    if args.targeted:
        utils.set_logger(os.path.join(experiment_path, 'beta_' + str(int(args.beta)) +  str(int(args.beta*10)) + '_target_'+str(args.target_label) + '_attack.log'))
    else:
        utils.set_logger(os.path.join(experiment_path, 'beta_' + str(int(args.beta)) +  str(int(args.beta*10)) + '_untarget_' + 'attack.log'))

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


    attack(model, loss_fn, test_dl, metrics, params, logger, beta=args.beta,
           num_steps=args.num_steps, targeted=args.targeted, target_label=args.target_label)
