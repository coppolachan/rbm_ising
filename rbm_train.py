###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
#
# (2018) 
# Guido Cossu <gcossu.work@gmail.com>
# Tommaso Giani
#
##############################################################


from __future__ import print_function

import sys
import json 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from tqdm import *

# PyTorch
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import rbm_pytorch

#####################################################
MNIST_SIZE = 784  # 28x28
#####################################################


def imgshow(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    Wmin = img.min
    Wmax = img.max
    plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)
    #plt.imsave(f, npimg)


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

print (json.dumps(parameters, indent=True))
args = parser.parse_args()
print(args)

print("Using library:", torch.__file__)
hidden_layers = parameters['hidden_size'] # note that you have to give the full dimension LxL if you are in 2 dim

# For the MNIST data set
if parameters['model'] == 'mnist':
    model_size = MNIST_SIZE
    image_size = 28
    dataset = datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=parameters['batch_size'], shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./DATA/MNIST_data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=parameters['batch_size'])
#############################
elif parameters['model'] == 'ising':
    # For the Ising Model data set
    model_size = parameters['ising_size']  # note that you have to give the full dimension LxL if you are in 2 dim
    image_size = parameters['ising_size']
    train_loader = torch.utils.data.DataLoader(rbm_pytorch.CSV_Ising_dataset(parameters['training_data'], size=model_size), shuffle=True,
                                               batch_size=parameters['batch_size'], drop_last=True)

# Read the model, example
rbm = rbm_pytorch.RBM(k=parameters['kCD'], n_vis=model_size, n_hid=hidden_layers)

# load the model, if the file is present
if parameters['ckpoint'] is not None:
    print("Loading saved network state from file", parameters['ckpoint'])
    rbm.load_state_dict(torch.load(parameters['ckpoint']))

##############################
# Training parameters

learning_rate = parameters['lrate']
mom = parameters['momentum']  # momentum
damp = 0.0  # dampening factor
wd = parameters['weight_decay']    # weight decay 

train_op = optim.SGD(rbm.parameters(), lr=learning_rate,
                     momentum=mom, dampening=damp, weight_decay=wd)

# progress bar
pbar = tqdm(range(parameters['start_epoch'], parameters['epochs']))

loss_file = open(parameters['text_output_dir'] + "Loss_timeline.data_" + str(parameters['model']) + "_lr" + str(learning_rate) + "_wd" + str(wd) + "_mom" + str(
    mom) + "_epochs" + str(parameters['epochs']), "w", buffering=1)

#Changed loss file output, changing headings here too

loss_file.write("# Parameters: " + json.dumps(parameters) +"\n")
loss_file.write("# Epoch |  Loss mean | reconstruction error mean | log free energy mean | logz | ll mean | ll error up | ll error down \n")

# Run the RBM training
for epoch in pbar:
    loss_ = []
    full_reconstruction_error = []
    free_energy_ = []
    data_size_ = []
    
    for i, (data, target) in enumerate(train_loader):
        data_input = Variable(data.view(-1, model_size))
        new_visible, hidden, h_prob, v_prob = rbm(data_input)

        # loss function: see Fisher eq 28 (Training RBM: an Introduction)
        # the average on the training set of the gradients is
        # the sum of the derivative averaged over the training set minus the average on the model
        # still possible instabilities here, so I am computing the gradient myself
        data_free_energy = rbm.free_energy_batch_mean(data_input)  # note: it does not include Z
        loss = data_free_energy - rbm.free_energy_batch_mean(new_visible)
        loss_.append(loss.data[0])
        free_energy_.append(data_input.size(0) * data_free_energy.data[0] )
        data_size_.append(data_input.size(0))
        
        reconstruction_error = rbm.loss(data_input, new_visible)
        full_reconstruction_error.append(reconstruction_error.data[0])

        # Update gradients
        train_op.zero_grad()
        # manually update the gradients, do not use autograd
        rbm.backward(data_input, new_visible)
        train_op.step()
    
    re_mean = np.mean(full_reconstruction_error)
    loss_mean = np.mean(loss_)

    data_size_sum = np.sum(data_size_)
    log_free_energy_mean = np.sum(free_energy_)/data_size_sum

    # Compute logz only once per 10 epochs
    if epoch % 10 == 0:
        logz , logz_up, logz_down = rbm.annealed_importance_sampling(1, 10000, 100)
        log_likelihood_mean = log_free_energy_mean - logz
        ll_error_up = (-logz_down + logz)
        ll_error_down = (-logz + logz_up)

        loss_file.write("%6d %15.5g %10.6f %15.5f %15.5f %15.5f %15.6f %15.6f \n" % 
             (epoch, loss_mean, re_mean, log_free_energy_mean, logz, log_likelihood_mean , ll_error_up, ll_error_down) )


    # Update the progress bar, note that the log_likelihood_mean is updated only every few epochs
    pbar.set_description("Epoch %4d - Loss %8.5f - RE %5.3g  LL %5.3g" % (epoch, loss_mean, re_mean, log_likelihood_mean))

    if epoch % 10 == 0:
        torch.save(rbm.state_dict(), parameters['text_output_dir'] + "trained_rbm.pytorch." + str(epoch))

# Save the final model
torch.save(rbm.state_dict(), parameters['text_output_dir'] + "trained_rbm.pytorch.last")
loss_file.close()
