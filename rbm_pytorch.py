###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
#
# RBM module
#
# 2017 Guido Cossu <gcossu.work@gmail.com>
#
##############################################################


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from tqdm import *

import argparse
import torch
import torch.utils.data
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp


class CSV_Ising_dataset(Dataset):
    """ Defines a CSV reader """
    def __init__(self, csv_file, size=32, transform=None):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype="float32")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)

class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500, k=5):
        super(RBM, self).__init__()
        # definition of the constructor
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.W = nn.Parameter(torch.zeros(n_hid, n_vis))
        #nn.init.uniform(self.W, a=-1, b=1)
        nn.init.normal(self.W,mean=0, std=0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.CDiter = k

    def sample_probability(self, prob, random):
        """Get samples from a tensor of probabilities.

            :param probs: tensor of probabilities
            :param rand: tensor (of the same shape as probs) of random values
            :return: binary sample of probabilities
        """
        return F.relu(torch.sign(prob - random))

    def hidden_from_visible(self, visible):
        # Enable or disable neurons depending on probabilities
        probability = torch.sigmoid(F.linear(visible, self.W, self.h_bias))
        random_field = Variable(torch.rand(probability.size()))
        new_states = self.sample_probability(probability, random_field)
        return new_states, probability

    def visible_from_hidden(self, hid):
        # Enable or disable neurons depending on probabilities
        probability = torch.sigmoid(F.linear(hid, self.W.t(), self.v_bias))
        random_field = Variable(torch.rand(probability.size()))
        new_states = self.sample_probability(probability, random_field)

        return new_states, probability

    def new_state(self, visible, use_prob=False):
        hidden, probhid = self.hidden_from_visible(visible)
        if (use_prob):
            new_visible, probvis = self.visible_from_hidden(probhid)
        else:
            new_visible, probvis = self.visible_from_hidden(hidden)
            
        # new_hidden, probhid_new = self.hidden_from_visible(probvis)

        return hidden, probhid, new_visible, probvis

    def forward(self, input):
        """
        Necessary function for Module classes
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.

        """
        # Contrastive divergence
        hidden, h_prob, vis, v_prob = self.new_state(input)
        
        for _ in range(self.CDiter-1):
            hidden, h_prob, vis, v_prob = self.new_state(vis, use_prob=False)
        
        return vis, hidden, h_prob, v_prob



    def loss(self, ref, test):
        return F.mse_loss(test, ref, size_average=True)

    def spin_flip(self, conf, s = 0):
        #print("s:",s)
        if (s > self.n_vis):
            return
        if (conf[0,s] == 0):
            conf[0,s] = 1
            return
        else:
            conf[0,s] = 0
            self.spin_flip(conf, s+1)

    def probabilities(self, Z):
        field = torch.zeros(1,self.n_vis)
        # fill with -1s
        f_conf = open('probrbm', 'w')
        for conf in xrange(2**self.n_vis-1):
            F = self.free_energy(Variable(field)).data.numpy()
            p = exp(-F)/Z
            f_conf.write(str(conf)+" "+ str(p) + " " + str(Z) + " " + str(F)+"\n")
            #print(field)
            self.spin_flip(field) 

        F = self.free_energy(Variable(field)).data.numpy()
        p = exp(-F)/Z
        f_conf.write(str(conf+1)+" "+ str(p) + " " + str(Z) + " " + str(F)+"\n")
        f_conf.close()

    def partition_function(self):
        # Sum the exp free energy over all visible states configurations
        field = torch.zeros(1,self.n_vis)
        # fill with -1s
        Z = 0.0
        for conf in xrange(2**self.n_vis-1):
            F = self.free_energy(Variable(field)).data.numpy()
            Z += exp(-F)
            self.spin_flip(field) 

        Z += exp(-self.free_energy(Variable(field)).data.numpy())
        return Z

    def free_energy(self, v):
        # computes log( p(v) )
        # eq 2.20 Asja Fischer
        # double precision
        vbias_term = v.mv(self.v_bias).double()  # = v*v_bias
        wx_b = F.linear(v, self.W, self.h_bias).double()  # = vW^T + h_bias
        
        # sum over the elements of the vector
        hidden_term = wx_b.exp().add(1).log().sum(1)  

        # notice that for batches of data the result is still a vector of size num_batches
        return -(hidden_term + vbias_term).mean()  # mean along the batches

    def backward(self, target, vk):
        # p(H_i | v) where v is the input data
        probability = torch.sigmoid(F.linear(target, self.W, self.h_bias))

        # p(H_i | v) where v is the negative visible layer
        h_prob_negative = torch.sigmoid(F.linear(vk, self.W, self.h_bias))
        # Update the W
        training_set_avg = probability.t().mm(target)
        # The minus sign comes from the implementation of the SGD in pytorch 
        # see http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
        # the learning rate has a negative sign in front
        self.W.grad = -(training_set_avg - h_prob_negative.t().mm(vk)) / probability.size(0)
        #print(self.W.grad)

        # Update the v_bias
        self.v_bias.grad = -(target - vk).mean(0)
        #print("vbias", self.v_bias)

        # Update the h_bias
        self.h_bias.grad = -(probability - h_prob_negative).mean(0)
        #print("hbias grad", self.h_bias.grad)
