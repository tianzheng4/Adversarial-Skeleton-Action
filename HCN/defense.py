# encoding: utf-8


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint, multinomial_proportions_confint

class NTU_DEFENSE:
    def __init__(self, mode='test', sigma=0.1, num_samples=100, filter_num=5):
        ## General
        self.num_frames = 64
        self.mode = mode
        self.sigma = sigma
        self.num_classes = 60
        self.fail_prob = 0.05
        self.num_samples = num_samples
        self.filter_num = filter_num

        self.pre_nodes = [1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, \
                                         15, 1, 17, 18, 19, 2, 8, 8, 12, 12]
        self.pre_nodes = [x-1 for x in self.pre_nodes]
        self.end_nodes = list(range(1, 25))

    def certification(self, model, data, label, data_adv=None, mask=None):
        if mask is None:
            bone_length = self.preprocess(data.data)
            mask = bone_length.abs().max(2)[0]
            mask = torch.sign(torch.max(mask - 1e-5, torch.zeros_like(mask))).float()
            mask = mask.view(data.size(0), 1, data.size(2), 1, data.size(4))
            mask = mask.repeat(1, data.size(1), 1, data.size(3), 1).cuda()

        counts = self.counts(data, model, mask).numpy()

        # max_counts = np.max(counts, axis=1)
        pred_label = np.argmax(counts, axis=1)

        p1, p2 = [], []
        for i in range(counts.shape[0]):
            p = multinomial_proportions_confint(np.sort(counts[i,:])[::-1], alpha=self.fail_prob)
            p1.append(p[0,0])
            p2.append(p[1,1])


        radius = np.maximum(0.5*self.sigma * (norm.ppf(np.array(p1)) - norm.ppf(np.array(p2))), 0.0)

        if data_adv is not None:
            l2_norm = self.L2_distance(data.data, data_adv.data).cpu().numpy()
            return np.logical_and(np.greater(radius, l2_norm), np.equal(pred_label, label.data.cpu().numpy()))
        return np.where(np.equal(pred_label, label.data.cpu().numpy()), radius, -1)

    ## input data is a variable
    def randomized_smoothing(self, model, data, mask=None):

        if mask is None:
            bone_length = self.preprocess(data.data)
            mask = bone_length.abs().max(2)[0]
            mask = torch.sign(torch.max(mask - 1e-5, torch.zeros_like(mask))).float()
            mask = mask.view(data.size(0), 1, data.size(2), 1, data.size(4))
            mask = mask.repeat(1, data.size(1), 1, data.size(3), 1).cuda()

        if self.mode == 'train':
            data.data += self.sigma*torch.randn_like(data.data)*mask
            return data

        elif self.mode == 'eval' or 'test':
            output = torch.zeros_like(model(data).data)
            output = Variable(output, requires_grad=False)

            for iter in range(self.num_samples):
                data_tmp = data.data.clone()

                noise = self.sigma*torch.randn_like(data.data)*mask
                data.data += noise
                if self.filter_num == 5:
                    data = self.gaussian_filter_5(data, mask=mask)
                else:
                    data = self.gaussian_filter_7(data, mask=mask)
                output.data += model(data).data

                data.data = data_tmp.clone()

            return output

    def randomized_smoothing_attack(self, model, data, mask=None):

        if mask is None:
            bone_length = self.preprocess(data.data)
            mask = bone_length.abs().max(2)[0]
            mask = torch.sign(torch.max(mask - 1e-5, torch.zeros_like(mask))).float()
            mask = mask.view(data.size(0), 1, data.size(2), 1, data.size(4))
            mask = mask.repeat(1, data.size(1), 1, data.size(3), 1).cuda()

        elif self.mode == 'eval' or 'test':
            output = torch.zeros_like(model(data).data)
            output = Variable(output, requires_grad=False)

            for iter in range(5):
                data_tmp = data.data.clone()

                noise = self.sigma*torch.randn_like(data.data)*mask
                data.data += noise
                data = self.gaussian_filter_5(data, mask=mask)
                output += model(data)

                data.data = data_tmp.clone()
            return output

    def randomized_smoothing_v2(self, model, data, mask=None):

        if mask is None:
            bone_length = self.preprocess(data.data)
            mask = bone_length.abs().max(2)[0]
            mask = torch.sign(torch.max(mask - 1e-5, torch.zeros_like(mask))).float()
            mask = mask.view(data.size(0), 1, data.size(2), 1, data.size(4))
            mask = mask.repeat(1, data.size(1), 1, data.size(3), 1).cuda()


        if self.mode == 'train':
            data.data += self.sigma*torch.randn_like(data.data)*mask
            return data

        elif self.mode == 'eval' or 'test':
            return self.counts(data, model, mask).float().cuda()

    def gaussian_filter_7(self, data, mask=None, device=None):

        if mask is None:
            print('no mask initialized')
            exit()

        shape = list(data.data.size())
        shape[2] += 6

        ## initialize aug_data and kernel
        if device is None:
            aug_data = torch.zeros(shape, dtype=torch.float32).cuda()
            kernel = torch.FloatTensor([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]).cuda()
        else:
            aug_data = torch.zeros(shape, dtype=torch.float32).cuda(device=device)
            kernel = torch.FloatTensor([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]).cuda(device=device)

        aug_data[:, :, 0:3, :] = data.data[:, :, 0:3, :].clone()
        aug_data[:, :, -3:, :] = data.data[:, :, -3:, :].clone()
        aug_data[:, :, 3:-3, :] = data.data.clone()

        ## compute data.data
        if device is None:
            data.data = torch.zeros_like(data.data, dtype=torch.float32).cuda()
        else:
            data.data = torch.zeros_like(data.data, dtype=torch.float32).cuda(device=device)

        for i in range(7):
            data.data += kernel[i] * aug_data[:, :, i:self.num_frames+i, :, :]

        data.data = data.data*mask
        return data

    def gaussian_filter_5(self, data, mask=None, device=None):

        shape = list(data.data.size())
        shape[2] += 6

        if device is None:
            aug_data = torch.zeros(shape, dtype=torch.float32).cuda()
            kernel = torch.FloatTensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136]).cuda()
        else:
            aug_data = torch.zeros(shape, dtype=torch.float32).cuda(device=device)
            kernel = torch.FloatTensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136]).cuda(device=device)

        aug_data[:, :, 0:3, :] = data.data[:, :, 0:3, :].clone()
        aug_data[:, :, -3:, :] = data.data[:, :, -3:, :].clone()
        aug_data[:, :, 3:-3, :] = data.data.clone()

        if device is None:
            data.data = torch.zeros_like(data.data, dtype=torch.float32).cuda()
        else:
            data.data = torch.zeros_like(data.data, dtype=torch.float32).cuda(device=device)

        for i in range(5):
            data.data += kernel[i] * aug_data[:, :, i:self.num_frames+i, :, :]

        data.data = data.data*mask
        return data

    def counts(self, data, model, mask=None):
        batch_size = int(data.size()[0])
        counts = torch.zeros((batch_size, self.num_classes), dtype=torch.int)

        data_tmp = data.data.clone()
        for iter in range(self.num_samples):

            noise = self.sigma*torch.randn_like(data.data)*mask
            data.data += noise

            # data = self.gaussian_filter_5(data, mask=mask)
            output = model(data).data

            ## output (max:1, others:0)

            counts += (torch.sign(torch.min(output - output.max(1)[0].unsqueeze(1) + 1e-6,
                                          0.0000*torch.zeros_like(output).float())) + 1).cpu().int()
            data.data = data_tmp.clone()

        return counts

    def L2_distance(self, data, data_adv):
        shape = list(data.size())
        return torch.sqrt(torch.sum((data_adv - data)**2, tuple(range(1, len(shape)))))

    def preprocess(self, data):

        bone_length = torch.norm(data[:, :, :, self.end_nodes, :] - data[:, :, :, self.pre_nodes, :], dim=1)

        return bone_length
