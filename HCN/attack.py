# encoding: utf-8


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class NTU_ADMM_ATTACK:
    def __init__(self, num_frames=64):

        ## General
        self.num_frames = num_frames

        ## Bones
        self.pre_nodes = [1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, \
                                         15, 1, 17, 18, 19, 2, 8, 8, 12, 12]
        self.pre_nodes = [x-1 for x in self.pre_nodes]
        self.end_nodes = list(range(1, 25)) # index starts from 0

        ##Joint angles
        self.angle_joints = {'waist': (1, 2, 21), 'chest1': (2, 21, 3), 'chest2': (5, 21, 9),
                             'neck': (21, 3, 4), 'base1': (2, 1, 13), 'base2': (2, 1, 17),
                             'shoulder1': (21, 9, 10), 'shoulder2': (21, 5, 6), 'elbow1': (9, 10, 11),
                             'elbow2': (5, 6, 7), 'forearm1': (10, 11, 12), 'forearm2': (6, 7, 8),
                             'wrist11': (11, 12, 24), 'wrist12': (11, 12, 25), 'wrist21': (7, 8, 22),
                             'wrist22': (7, 8, 23), 'hip1': (1, 13, 14), 'hip2': (1, 17, 18),
                             'knee1': (13, 14, 15), 'knee2': (17, 18, 19), 'ankle1': (14, 15, 16),
                             'ankle2': (18, 19, 20)} ## 22 joint angles
        ## -1 because index starts from 0
        self.pre_joints = [self.angle_joints[k][0]-1 for k in self.angle_joints]
        self.middle_joints = [self.angle_joints[k][1]-1 for k in self.angle_joints]
        self.end_joints = [self.angle_joints[k][2]-1 for k in self.angle_joints]

    def preprocess(self, data):

        bone_length = torch.norm(data[:, :, :, self.end_nodes, :] - data[:, :, :, self.pre_nodes, :], dim=1)

        joint_bone1 = torch.norm(data[:, :, :, self.pre_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)
        joint_bone2 = torch.norm(data[:, :, :, self.end_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)

        velocity = data[:, :, 1:self.num_frames, :, :] - data[:, :, 0:self.num_frames-1, :, :]

        return bone_length, joint_bone1, joint_bone2, velocity

    def bone_constraints(self, data_adv, bone_length, epsilon=0.01, mask=None):

        mask_bone = mask[:, 0, :, self.pre_nodes, :]

        relative_pos_adv = data_adv[:, :, :, self.end_nodes, :] - data_adv[:, :, :, self.pre_nodes, :]

        return F.relu(mask_bone*(torch.abs(torch.norm(relative_pos_adv, dim=1) - bone_length))/(bone_length+1e-5) - epsilon)

    def angle_constraints(self, data, data_adv, bone1, bone2, epsilon=0.1, mask=None):

        mask_theta = mask.data[:,0,:,self.pre_joints,:]

        theta1 = mask_theta*torch.norm(data_adv[:, :, :, self.pre_joints, :] - data[:, :, :, self.pre_joints, :], dim=1)/(bone1 + 1e-5)
        theta2 = mask_theta*torch.norm(data_adv[:, :, :, self.end_joints, :] - data[:, :, :, self.end_joints, :], dim=1)/(bone2 + 1e-5)

        theta3 = mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone1 + 1e-5) + \
                  mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone2 + 1e-5)

        ## from inequality to equality
        return F.relu((theta1 + theta2 + theta3) - epsilon)

    def temporal_smoothness_constraints(self, data_adv, velocity, epsilon=0.1, mask=None):

        mask_vel = mask[:, 0, 1:self.num_frames, :, :]

        velocity_adv = data_adv[:, :, 1:self.num_frames, :, :] - data_adv[:, :, 0:self.num_frames-1, :, :]

        return F.relu(mask_vel*torch.norm(velocity_adv - velocity, dim=1)/(torch.norm(velocity, dim=1) + 1e-4) - epsilon)

    def bone_vio_rate(self, data_adv, bone_length, mask=None):

        mask_bone = mask[:, 0, :, self.pre_nodes, :]

        relative_pos_adv = data_adv[:, :, :, self.end_nodes, :] - data_adv[:, :, :, self.pre_nodes, :]

        return torch.sum(mask_bone*(torch.abs(torch.norm(relative_pos_adv, dim=1) - bone_length))/(bone_length+1e-5))/torch.sum(mask_bone)

    def angle_vio_rate(self, data, data_adv, bone1, bone2, mask=None):

        mask_theta = mask.data[:,0,:,self.pre_joints,:]

        theta1 = mask_theta*torch.norm(data_adv[:, :, :, self.pre_joints, :] - data[:, :, :, self.pre_joints, :], dim=1)/(bone1 + 1e-5)
        theta2 = mask_theta*torch.norm(data_adv[:, :, :, self.end_joints, :] - data[:, :, :, self.end_joints, :], dim=1)/(bone2 + 1e-5)

        theta3 = mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone1 + 1e-5) + \
                  mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone2 + 1e-5)

        ## from inequality to equality
        return torch.sum(theta1 + theta2 + theta3)/torch.sum(mask_theta)

    def kinetics_vio_rate(self, data_adv, velocity):

        velocity_adv = data_adv[:, :, 1:self.num_frames, :, :] - data_adv[:, :, 0:self.num_frames-1, :, :]

        return torch.mean(torch.abs(torch.sum(velocity_adv**2, dim=(1, 2, 3, 4))/torch.sum(velocity**2, dim=(1, 2, 3, 4)) - 1.0))

    def L2_distance(self, data, data_adv):
        shape = list(data.size())
        return torch.mean(torch.sqrt(torch.sum((data_adv - data)**2, tuple(range(1, len(shape))))))

class Kinetics_ADMM_ATTACK:
    def __init__(self, num_frames=64):

        ## General
        self.num_frames = num_frames

        ## Bones
        self.bones = {'b1': (1, 2), 'b2': (2, 3), 'b3': (3, 4),
                             'b4': (1, 5), 'b5': (5, 6), 'b6': (6, 7),
                             'b7': (1, 8), 'b8': (8, 9), 'b9': (9, 10),
                             'b10': (1, 11), 'b11': (11, 12), 'b12': (12, 13),
                             'b13': (0, 1), 'b14': (0, 14), 'b15': (0, 15),
                             'b16': (14, 16), 'b17': (15, 17)
                             # , 'b18': (14, 16), 'b19': (15, 17)
                             }
        ## not -1 because index already starts from 0
        self.pre_nodes = [self.bones[k][0] for k in self.bones]
        self.end_nodes = [self.bones[k][1] for k in self.bones]

        ##Joint angles
        self.angle_joints = {'j1': (1, 2, 3), 'j2': (2, 3, 4), 'j3': (1, 5, 6),
                             'j4': (1, 8, 9), 'j5': (8, 9, 10), 'j6': (1, 11, 12),
                             'j7': (11, 12, 13), 'j8': (16, 0, 17), 'j9': (14, 0, 15),
                             'j10': (14, 0, 16), 'j11': (15, 0, 17)} ## 9 joint angles
        ## not -1 because index already starts from 0
        self.pre_joints = [self.angle_joints[k][0] for k in self.angle_joints]
        self.middle_joints = [self.angle_joints[k][1] for k in self.angle_joints]
        self.end_joints = [self.angle_joints[k][2] for k in self.angle_joints]

    def preprocess(self, data):

        bone_length = torch.norm(data[:, :, :, self.end_nodes, :] - data[:, :, :, self.pre_nodes, :], dim=1)

        joint_bone1 = torch.norm(data[:, :, :, self.pre_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)
        joint_bone2 = torch.norm(data[:, :, :, self.end_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)

        velocity = data[:, :, 1:self.num_frames, :, :] - data[:, :, 0:self.num_frames-1, :, :]

        return bone_length, joint_bone1, joint_bone2, velocity

    def bone_constraints(self, data_adv, bone_length, epsilon=0.01, mask=None):

        mask_bone = mask[:, 0, :, self.pre_nodes, :]

        relative_pos_adv = data_adv[:, :, :, self.end_nodes, :] - data_adv[:, :, :, self.pre_nodes, :]

        return F.relu(mask_bone*(torch.abs(torch.norm(relative_pos_adv, dim=1) - bone_length))/(bone_length+1e-3) - epsilon)

    def angle_constraints(self, data, data_adv, bone1, bone2, epsilon=0.1, mask=None):

        mask_theta = mask.data[:,0,:,self.pre_joints,:]

        theta1 = mask_theta*torch.norm(data_adv[:, :, :, self.pre_joints, :] - data[:, :, :, self.pre_joints, :], dim=1)/(bone1 + 1e-3)
        theta2 = mask_theta*torch.norm(data_adv[:, :, :, self.end_joints, :] - data[:, :, :, self.end_joints, :], dim=1)/(bone2 + 1e-3)

        theta3 = mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone1 + 1e-3) + \
                  mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone2 + 1e-3)

        ## from inequality to equality
        return F.relu((theta1 + theta2 + theta3) - epsilon)

    def temporal_smoothness_constraints(self, data_adv, velocity, epsilon=0.1, mask=None):

        mask_vel = mask[:, 0, 1:self.num_frames, :, :]

        velocity_adv = data_adv[:, :, 1:self.num_frames, :, :] - data_adv[:, :, 0:self.num_frames-1, :, :]

        return F.relu(mask_vel*torch.norm(velocity_adv - velocity, dim=1)/(torch.norm(velocity, dim=1) + 1e-3) - epsilon)


    def bone_vio_rate(self, data_adv, bone_length, mask=None):

        mask_bone = mask[:, 0, :, self.pre_nodes, :]

        relative_pos_adv = data_adv[:, :, :, self.end_nodes, :] - data_adv[:, :, :, self.pre_nodes, :]

        return torch.sum(mask_bone*(torch.abs(torch.norm(relative_pos_adv, dim=1) - bone_length))/(bone_length+1e-3))/torch.sum(mask_bone)

    def angle_vio_rate(self, data, data_adv, bone1, bone2, mask=None):

        mask_theta = mask.data[:,0,:,self.pre_joints,:]

        theta1 = mask_theta*torch.norm(data_adv[:, :, :, self.pre_joints, :] - data[:, :, :, self.pre_joints, :], dim=1)/(bone1 + 1e-3)
        theta2 = mask_theta*torch.norm(data_adv[:, :, :, self.end_joints, :] - data[:, :, :, self.end_joints, :], dim=1)/(bone2 + 1e-3)

        theta3 = mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone1 + 1e-3) + \
                  mask_theta*torch.norm(data_adv[:, :, :, self.middle_joints, :] - data[:, :, :, self.middle_joints, :], dim=1)/(bone2 + 1e-3)

        ## from inequality to equality
        return torch.sum(theta1 + theta2 + theta3)/torch.sum(mask_theta)

    def kinetics_vio_rate(self, data_adv, velocity):

        velocity_adv = data_adv[:, :, 1:self.num_frames, :, :] - data_adv[:, :, 0:self.num_frames-1, :, :]

        return torch.mean(torch.abs(torch.sum(velocity_adv**2, dim=(1, 2, 3, 4))/(torch.sum(velocity**2, dim=(1, 2, 3, 4))+1e-2) - 1.0))

    def L2_distance(self, data, data_adv):
        shape = list(data.size())
        return torch.mean(torch.sqrt(torch.sum((data_adv - data)**2, tuple(range(1, len(shape))))))
