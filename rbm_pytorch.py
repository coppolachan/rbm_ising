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
from tqdm import *
import argparse

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp

# From PyDeep
def log_sum_exp(x, axis=0):
    """ Calculates the logarithm of the sum of e to the power of input 'x'. The method tries to avoid \
        overflows by using the relationship: log(sum(exp(x))) = alpha + log(sum(exp(x-alpha))).

    :param x: data.
    :type x: float or numpy array

    :param axis: Sums along the given axis.
    :type axis: int

    :return: Logarithm of the sum of exp of x.
    :rtype: float or numpy array.
    """
    alpha = x.max(axis) - np.log(np.finfo(np.float64).max) / 2.0
    if axis == 1:
        return np.squeeze(alpha + np.log(np.sum(np.exp(x.T - alpha), axis=0)))
    else:
        return np.squeeze(alpha + np.log(np.sum(np.exp(x - alpha), axis=0)))


def log_diff_exp(x, axis=0):
    """ Calculates the logarithm of the diffs of e to the power of input 'x'. The method tries to avoid \
        overflows by using the relationship: log(diff(exp(x))) = alpha + log(diff(exp(x-alpha))).

    :param x: data.
    :type x: float or numpy array

    :param axis: Diffs along the given axis.
    :type axis: int

    :return: Logarithm of the diff of exp of x.
    :rtype: float or numpy array.
    """
    alpha = x.max(axis) - np.log(np.finfo(np.float64).max) / 2.0
    #print("alpha", alpha)
    if axis == 1:
        return np.squeeze(alpha + np.log(np.diff(np.exp(x.T - alpha), n=1, axis=0)))
    else:
        return np.squeeze(alpha + np.log(np.diff(np.exp(x - alpha), n=1, axis=0)))



class CSV_Ising_dataset(Dataset):
    """ Defines a CSV reader """
    def __init__(self, csv_file, size=32, transform=None, skiprows=1):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",", skiprows=skiprows, dtype="float32")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("# Loaded training set of %d states" % self.datasize)

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

    def hidden_from_visible(self, visible, beta = 1.0):
        # Enable or disable neurons depending on probabilities
        activation = F.linear(visible, self.W, self.h_bias)
        if beta is not None:
            activation *= beta
        probability = torch.sigmoid(activation)
        random_field = Variable(torch.rand(probability.size()))
        new_states = self.sample_probability(probability, random_field)
        return new_states, probability

    def visible_from_hidden(self, hid, beta = 1.0):
        # Enable or disable neurons depending on probabilities
        activation = F.linear(hid, self.W.t(), self.v_bias)
        if beta is not None:
            activation *= beta
        probability = torch.sigmoid(activation)
        random_field = Variable(torch.rand(probability.size()))
        new_states = self.sample_probability(probability, random_field)

        return new_states, probability

    def new_state(self, visible, use_prob=False, beta = 1.0):
        hidden, probhid = self.hidden_from_visible(visible)
        if (use_prob):
            new_visible, probvis = self.visible_from_hidden(probhid, beta)
        else:
            new_visible, probvis = self.visible_from_hidden(hidden, beta)
            
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
            F = self.free_energy_batch_mean(Variable(field)).data.numpy()
            p = exp(-F)/Z
            f_conf.write(str(conf)+" "+ str(p) + " " + str(Z) + " " + str(F)+"\n")
            #print(field)
            self.spin_flip(field) 

        F = self.free_energy_batch_mean(Variable(field)).data.numpy()
        p = exp(-F)/Z
        f_conf.write(str(conf+1)+" "+ str(p) + " " + str(Z) + " " + str(F)+"\n")
        f_conf.close()

    def partition_function(self):
        # Sum the exp free energy over all visible states configurations
        # use only with an RMB with a limited number of states
        field = torch.zeros(1,self.n_vis)
        # fill with -1s
        Z = 0.0
        for conf in xrange(2**self.n_vis-1):
            F = self.free_energy_batch_mean(Variable(field)).data.numpy()
            Z += exp(-F)
            self.spin_flip(field) 

        Z += exp(-self.free_energy_batch_mean(Variable(field)).data.numpy())
        return Z

    def annealed_importance_sampling(self, k = 1, betas = 10000, num_chains = 100):
        """
        Approximates the partition function for the given model using annealed importance sampling.

            .. see also:: Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing \
               http://arxiv.org/pdf/1412.8566.pdf

        :param num_chains: Number of AIS runs.
        :type num_chains: int

        :param k: Number of Gibbs sampling steps.
        :type k: int

        :param betas: Number or a list of inverse temperatures to sample from.
        :type betas: int, numpy array [num_betas]
        """
        
        # Set betas
        if np.isscalar(betas):
            betas = np.linspace(0.0, 1.0, betas)
        
        # Start with random distribution beta = 0
        v = Variable(torch.sign(torch.rand(num_chains,self.n_vis)-0.5), volatile = True)  
        v = F.relu(v) # v in {0,1} and distributed randomly

        # Calculate the unnormalized probabilties of v
        # HERE: need another function that does not average across batches....
        lnpv_sum = -self.free_energy(v, betas[0])  #  denominator

        for beta in betas[1:betas.shape[0] - 1]:
            # Calculate the unnormalized probabilties of v
            lnpv_sum += self.free_energy(v, beta)

           # Sample k times from the intermidate distribution
            for _ in range(0, k):
                h, ph, v, pv = self.new_state(v, beta=beta)

            # Calculate the unnormalized probabilties of v
            lnpv_sum -= self.free_energy(v, beta)

        # Calculate the unnormalized probabilties of v
        lnpv_sum += self.free_energy(v, betas[betas.shape[0] - 1])

        lnpv_sum = np.float128(lnpv_sum.data.numpy())
        #print("lnpvsum", lnpv_sum)

        # Calculate an estimate of logz . 
        logz = log_sum_exp(lnpv_sum) - np.log(num_chains)

        # Calculate +/- 3 standard deviations
        lnpvmean = np.mean(lnpv_sum)
        lnpvstd = np.log(np.std(np.exp(lnpv_sum - lnpvmean))) + lnpvmean - np.log(num_chains) / 2.0
        lnpvstd = np.vstack((np.log(3.0) + lnpvstd, logz))
        #print("lnpvstd", lnpvstd)
        #print("lnpvmean", lnpvmean)
        #print("logz", logz)

        # Calculate partition function of base distribution
        baselogz = self.log_partition_function_infinite_temperature()

        # Add the base partition function
        logz = logz + baselogz
        logz_up = log_sum_exp(lnpvstd) + baselogz
        logz_down = log_diff_exp(lnpvstd) + baselogz

        return logz , logz_up, logz_down

    def log_partition_function_infinite_temperature(self):
        # computes log ( p(v) ) for random states
        return (self.n_vis) * np.log(2.0)

    def free_energy(self, v, beta=1.0):
        # computes log( p(v) )
        # eq 2.20 Asja Fischer
        # double precision
        
        vbias_term = v.mv(self.v_bias).double()  # = v*v_bias
        wx_b = F.linear(v, self.W, self.h_bias).double()  # = vW^T + h_bias
        
        # sum over the elements of the vector
        hidden_term = wx_b.exp().add(1).log().sum(1)  

        # notice that for batches of data the result is still a vector of size num_batches
        return (hidden_term + vbias_term)*beta  # mean along the batches

    def free_energy_batch_mean(self, v, beta = 1.0):
        return self.free_energy(v,beta).mean()

    def free_energy_batch_sum(self, v, beta = 1.0):
        return self.free_energy(v,beta).sum()

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

        # Update the v_bias
        self.v_bias.grad = -(target - vk).mean(0)

        # Update the h_bias
        self.h_bias.grad = -(probability - h_prob_negative).mean(0)
