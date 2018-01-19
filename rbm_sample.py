###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
#
#
# 2017 Guido Cossu <gcossu.work@gmail.com>
#
##############################################################


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from tqdm import *

import argparse
import torch
import torch.utils.data
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp, sqrt

import json
from pprint import pprint

import rbm_pytorch
import pandas as pd

#####################################################
MNIST_SIZE = 784  # 28x28
#####################################################

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])  
    return X_resample


def get_ising_variables(field, sign=-1):
    """ Get the Ising variables {-1,1} representation
    of the RBM Markov fields

    :param field: the RBM state (visible or hidden), numpy

    :param sign: sign of the conversion 

    :return: the Ising field

    """
    sign_field = np.full(field.shape, sign)

    return (2.0 * field + sign_field).astype(int)


def ising_magnetization(field):
    m = np.abs((field).mean())
    return np.array([m, m * m])

def ising_averages(mag_history, model_size, label=""):
    # Bootstrap samples
    resample_size = 50000
    mag_resample = bootstrap_resample(mag_history[:,0], n=resample_size)
    #print('original mean:', mag_history[:,0].mean() )
    #print('resampled mean:', mag_resample.mean() )
    #print('resampled error:', mag_resample.std()/sqrt(resample_size) )

    mag_err = model_size*((mag_history[:,0]-mag_history[:,0].mean())**2)
    susc_resample = bootstrap_resample(mag_err, n=resample_size)
    #print('original susc mean:', mag_err.mean() )
    #print('resampled susc mean:', susc_resample.mean() )
    #print('resampled susc error:', susc_resample.std()/sqrt(resample_size) )

    print(label, " ::: Magnetization: ", mag_resample.mean(), " +- ", mag_resample.std()/sqrt(resample_size), " - Susceptibility:", susc_resample.mean() , " +- ", susc_resample.std()/sqrt(resample_size))
    #plt.plot(mag_history[:,0], linewidth=0.2)
    #plt.show()


def imgshow(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    Wmin = img.min
    Wmax = img.max
    plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)


def sample_probability(prob, random):
    """Get samples from a tensor of probabilities.

        :param probs: tensor of probabilities
        :param rand: tensor (of the same shape as probs) of random values
        :return: binary sample of probabilities
    """
    torchReLu = nn.ReLU()
    return torchReLu(torch.sign(prob - random)).data


def hidden_from_visible(visible, W, h_bias):
    # Enable or disable neurons depending on probabilities
    probability = torch.sigmoid(F.linear(visible, W, h_bias))
    random_field = torch.rand(probability.size())
    new_states = sample_probability(probability, random_field)
    return new_states, probability


def visible_from_hidden(hid, W, v_bias):
    # Enable or disable neurons depending on probabilities
    probability = torch.sigmoid(F.linear(hid, W.t(), v_bias))
    random_field = torch.rand(probability.size())
    new_states = sample_probability(probability, random_field)
    return new_states, probability


def sample_from_rbm(steps, model, image_size, nstates=30, v_in=None):
    """ Samples from the RBM distribution function 

        :param steps: Number of Gibbs sampling steps.
        :type steps: int

        :param model: Trained RBM model.
        :type model: RBM class

        :param image_size: Linear size of output images
        :type image_size: int

        :param nstates: Number of states to generate concurrently
        :type nstates: int

        :param v_in: Initial states (optional)

        :return: Last generated visible state
    """

    if (parameters['initialize_with_training']):
        v = v_in
    else:
        # Initialize with zeroes
        v = torch.zeros(nstates, model.v_bias.data.shape[0])
        # Random initial visible state
        #v = F.relu(torch.sign(torch.rand(nstates,model.v_bias.data.shape[0])-0.5)).data

    v_prob = v
    
    magv = []
    magh = []
    # Run the Gibbs sampling for a number of steps
    for s in xrange(steps):
        #r = np.random.random()
        #if (r > 0.5):
        #    vin = torch.zeros(nstates, model.v_bias.data.shape[0])
        #    vin = torch.ones(nstates, model.v_bias.data.shape[0])
        #else:


        if (s % parameters['save interval'] == 0):
            if parameters['output_states']:
                imgshow(parameters['image_dir'] + "dream" + str(s),
                        make_grid(v.view(-1, 1, image_size, image_size)))
            else:
                imgshow(parameters['image_dir'] + "dream" + str(s),
                        make_grid(v_prob.view(-1, 1, image_size, image_size)))
            if args.verbose:
                print(s, "OK")

        # Update k steps
        #for _ in xrange(200):
        h, h_prob = hidden_from_visible(v, model.W.data, model.h_bias.data)
        v, v_prob = visible_from_hidden(h, model.W.data, model.v_bias.data)
        #vin = v
        
        
            # Save data
        if (s > parameters['thermalization']):
            magv.append(ising_magnetization(get_ising_variables(v.numpy())))
            magh.append(ising_magnetization(get_ising_variables(h.numpy())))
    return v, np.asarray(magv), np.asarray(magh)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
                    type=str)
parser.add_argument('--verbose', dest='verbose', default=False, help='Verbosity control',
                    type=bool, choices=[False, True])

args = parser.parse_args()
try:
    parameters = json.load(open(args.input_json))
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

if args.verbose:
    print(args)
    pprint(parameters)

hidden_layers = parameters['hidden_layers'] * parameters['hidden_layers']


# For the MNIST data set
if parameters['model'] == 'mnist':
    model_size = MNIST_SIZE
    image_size = 28
    if parameters['initialize_with_training']:
        print("Loading MNIST training set...")
        train_loader = datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
#############################
elif parameters['model'] == 'ising':
    # For the Ising Model data set
    image_size = parameters['ising']['size']
    model_size = image_size * image_size
    if parameters['initialize_with_training']:
        print("Loading Ising training set...")
        train_loader = rbm_pytorch.CSV_Ising_dataset(
            parameters['ising']['train_data'], size=model_size)

        # Compute magnetization and susceptibility
        train_mag = []
        for i in xrange(len(train_loader)):
            data = train_loader[i][0].view(-1, model_size)
            train_mag.append(ising_magnetization(get_ising_variables(data.numpy())))
        tr_magarray= np.asarray(train_mag)
        ising_averages(tr_magarray, model_size, "training_set")

# Read the model, example
rbm = rbm_pytorch.RBM(n_vis=model_size, n_hid=hidden_layers)

# load the model, if the file is present
try:
    print("Loading saved RBM network state from file",
          parameters['checkpoint'])
    rbm.load_state_dict(torch.load(parameters['checkpoint']))
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

print('Model succesfully loaded')

if parameters['initialize_with_training']:
    data = torch.zeros(parameters['concurrent samples'], model_size)
    for i in xrange(parameters['concurrent samples']):
        data[i] = train_loader[i + 100][0].view(-1, model_size)
    v, magv, magh = sample_from_rbm(parameters['steps'], rbm, image_size, v_in=data)
else:
    v, magv, magh = sample_from_rbm(
        parameters['steps'], rbm, image_size, parameters['concurrent samples'])


# Print statistics
ising_averages(magv, model_size, "v")
ising_averages(magh, model_size, "h")

logz, logz_up, logz_down = rbm.annealed_importance_sampling(k=1, betas = 10000, num_chains = 200)
print("LogZ ", logz, logz_up, logz_down)

# Save data
np.savetxt("Mag_history", magv)




"""

Example of JSON input

{
    "model": "ising",
    "checkpoint": "trained_rbm.pytorch.last",
    "image_dir": "./DATA/Dream_Ising/",
    "hidden_layers": 20,
    "steps": 500,
    "save interval": 2,
    "concurrent samples": 30,
    "initialize_with_training": false, 
    "output_states": true,
    "ising": {
        "train_data": "state1.data",
        "size": 32
    }
}


"""
