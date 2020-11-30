# encoding: utf-8

"""
@author: huguyuehuhu
@time: 18-4-12 下午3:10
Permission is given to modify the code, any problem please contact huguyuehuhu@gmail.com
"""
import torch
from feeder.feeder import Feeder
import numpy as np

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev
    """
    if 'NTU-RGB-D' in params.dataset_name :
        if 'CV' in params.dataset_name:
            params.train_feeder_args["data_path"] = params.dataset_dir+'/xview/train_data_joint.npy'
            params.train_feeder_args["num_frame_path"] = params.dataset_dir+'/xview/train_num_frame.npy'
            params.train_feeder_args["label_path"] = params.dataset_dir + '/xview/train_label.pkl'
            params.test_feeder_args["data_path"] = params.dataset_dir + '/xview/val_data_joint.npy'
            params.test_feeder_args["num_frame_path"] = params.dataset_dir +'/xview/val_num_frame.npy'
            params.test_feeder_args["label_path"] = params.dataset_dir + '/xview/val_label.pkl'

        if 'CS' in params.dataset_name:
            params.train_feeder_args["data_path"] = params.dataset_dir + '/xsub/train_data_joint.npy'
            params.train_feeder_args["num_frame_path"] = params.dataset_dir + '/xsub/train_num_frame.npy'
            params.train_feeder_args["label_path"] = params.dataset_dir + '/xsub/train_label.pkl'
            params.test_feeder_args["data_path"]= params.dataset_dir + '/xsub/val_data_joint.npy'
            params.test_feeder_args["num_frame_path"] = params.dataset_dir + '/xsub/val_num_frame.npy'
            params.test_feeder_args["label_path"] = params.dataset_dir + '/xsub/val_label.pkl'

    elif 'Kinetics' in params.dataset_name:
        params.train_feeder_args["data_path"] = params.dataset_dir + '/kinetics-2/train_data_joint.npy'
        params.train_feeder_args["num_frame_path"] = params.dataset_dir + '/kinetics-2/train_num_frame.npy'
        params.train_feeder_args["label_path"] = params.dataset_dir + '/kinetics-2/train_label.pkl'
        params.test_feeder_args["data_path"]= params.dataset_dir + '/kinetics-2/val_data_joint.npy'
        params.test_feeder_args["num_frame_path"] = params.dataset_dir + '/kinetics-2/val_num_frame.npy'
        params.test_feeder_args["label_path"] = params.dataset_dir + '/kinetics-2/val_label.pkl'



    if types == 'train':
        if not hasattr(params,'batch_size_train'):
            params.batch_size_train = params.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params.train_feeder_args),
            batch_size=params.batch_size_train,
            shuffle=True,
            num_workers=params.num_workers,pin_memory=params.cuda)

    if types == 'test':
        if not hasattr(params,'batch_size_test'):
            params.batch_size_test = params.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params.test_feeder_args),
            batch_size=params.batch_size_test ,
            shuffle=False,
            num_workers=params.num_workers,pin_memory=params.cuda)

    return loader

if __name__ == '__main__':

    pass
