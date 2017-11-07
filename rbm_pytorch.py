###############################################################
#
# Restricted Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
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

#####################################################
MNIST_SIZE = 784  # 28x28
#####################################################

# Define a CSV reader dataset
class CSV_Ising_dataset(Dataset):
    def __init__(self, csv_file, size=32, transform=None):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",",
                             skiprows=1, dtype="float32")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size * size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)

# Define a reader for npz files
class Numpy_Ising_dataset(Dataset):
    def __init__(self, npy_file, size=32, transform=None):
        self.npy_file = npy_file
        self.size = size
        isingdata = np.load(npy_file)
        print(np.asarray(isingdata))
        self.imgs = torch.from_numpy(isingdata.reshape(-1, size * size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)


def imgshow(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    Wmin = img.min
    Wmax = img.max
    # plt.imshow(npimg)
    plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)


class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500, k=5):
        super(RBM, self).__init__()
        # definition of the constructor
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
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

    def new_state(self, visible):
        hidden, probhid = self.hidden_from_visible(visible)
        new_visible, probvis = self.visible_from_hidden(probhid)
        # new_hidden, probhid_new = self.hidden_from_visible(probvis)

        return hidden, probhid, new_visible, probvis

    def forward(self, input):
        """
        Necessary function for Module classes
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.

        """
        hidden, h_prob, new_vis, v_prob = self.new_state(input)

        # Contrastive divergence
        for _ in range(self.CDiter):
            hidden, h_prob, new_vis, v_prob = self.new_state(new_vis)

        return new_vis, hidden, h_prob, v_prob

    def loss(self, ref, test):
        # mseloss = torch.nn.MSELoss()# not ok, it will check for gradients
        # difference in prediction
        return F.mse_loss(test, ref, size_average=True)

    def free_energy(self, v):
        # computes -log( p(v) )
        # eq 2.20 Asja
        # double precision
        vbias_term = v.mv(self.v_bias).double()  # = v*v_bias
        wx_b = F.linear(v, self.W, self.h_bias).double()  # = vW^T + h_bias
        hidden_term = wx_b.exp().add(1).log().sum(1)  # sum over the elements of the vector

        # notice that for batches of data the result is still a vector of size num_batches
        return (-hidden_term - vbias_term).mean()  # mean along the batches

    def backward(self, target, v, h_prob):
        # vbias_term = v.mv(self.v_bias)  # = v*v_bias
        # pv = (F.linear(v, self.W, self.h_bias)).exp().add(1).prod(1)*vbias_term

        # p(H_i | v) where v is the input data
        probability = torch.sigmoid(F.linear(target, self.W, self.h_bias))

        # Update the W
        training_set_avg = probability.t().mm(target)
        self.W.grad = -(training_set_avg - h_prob.t().mm(v)) / \
            probability.size(0)

        # Update the v_bias
        # pv_v = v.t().mv(pv)
        self.v_bias.grad = -(target - v).mean(0)

        # Update the h_bias
        # pv_hp = h_prob.t().mv(pv) # p(H_i | v) * v  where the probability is given v as a machine state
        self.h_bias.grad = -(probability - h_prob).mean(0)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', dest='model', default='mnist', help='choose the model',
                    type=str, choices=['mnist', 'ising'])
parser.add_argument('--ckpoint', dest='ckpoint', help='pass a saved state',
                    type=str)
parser.add_argument('--epochs', dest='epochs', default=10, help='number of epochs',
                    type=int)
parser.add_argument('--batch', dest='batches', default=128, help='batch size',
                    type=int)
parser.add_argument('--hidden', dest='hidden_size', default=500, help='hidden feature size',
                    type=int)
parser.add_argument('--ising_size', dest='ising_size', default=32, help='lattice size for this Ising 2D model',
                    type=int)
parser.add_argument('--k', dest='kCD', default=2, help='number of Contrastive Divergence steps',
                    type=int)
parser.add_argument('--imgout', dest='image_output_dir', default='./', help='directory in which to save output images',
                    type=str)
parser.add_argument('--train', dest='training_data', default='state0.data', help='path to training input data',
                    type=str)
parser.add_argument('--txtout', dest='text_output_dir', default='./', help='directory in which to save text output data',
                    type=str)

args = parser.parse_args()
print(args)
print("Using library:", torch.__file__)
hidden_layers = args.hidden_size * args.hidden_size

# For the MNIST data set
if args.model == 'mnist':
    model_size = MNIST_SIZE
    image_size = 28
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batches, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./DATA/MNIST_data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batches)
#############################
elif args.model == 'ising':
    # For the Ising Model data set
    model_size = args.ising_size * args.ising_size
    image_size = args.ising_size
    train_loader = torch.utils.data.DataLoader(CSV_Ising_dataset(args.training_data, size=args.ising_size), shuffle=True,
                                               batch_size=args.batches)

# Read the model, example
rbm = RBM(k=args.kCD, n_vis=model_size, n_hid=hidden_layers)

# load the model, if the file is present
if args.ckpoint is not None:
    print("Loading saves network state from file", args.ckpoint)
    rbm.load_state_dict(torch.load(args.ckpoint))

##############################
# Training parameters

learning_rate = 0.1
mom = 0.9  # momentum
damp = 0.0  # dampening factor
wd = 0.0  # weight decay

train_op = optim.SGD(rbm.parameters(), lr=learning_rate,
                     momentum=mom, dampening=damp, weight_decay=wd)

# progress bar
pbar = tqdm(range(args.epochs))

loss_file = open(args.text_output_dir + "Loss_timeline.data_" + str(args.model) + "_lr" + str(learning_rate) + "_wd" + str(wd) + "_mom" + str(
    mom) + "_epochs" + str(args.epochs), "w")

# Run the RBM training
for epoch in pbar:
    loss_ = []

    for i, (data, target) in enumerate(train_loader):
        data_input = Variable(data.view(-1, model_size))
        # how to randomize?
        new_visible, hidden, h_prob, v_prob = rbm(data_input)

        # loss function: see Fisher eq 28 (Training RBM: an Introduction)
        # the average on the training set of the gradients is
        # the sum of the derivative averaged over the training set minus the average on the model
        # still having some instabilities here
        log_likelihood = rbm.free_energy(data_input)
        loss = log_likelihood - rbm.free_energy(new_visible)  # correct!
        loss_.append(loss.data[0])

        loss_file.write(
            str(i) + "\t" + str(epoch) + "\t" + str(epoch * (args.batches - 1) + i) + "\t" + str(loss.data[0]) + "\t" + str(log_likelihood.data[0]) + "\n")

        # Update gradients
        train_op.zero_grad()
        # loss.backward()
        rbm.backward(data_input, new_visible, h_prob)
        train_op.step()

    loss_mean = np.mean(loss_)
    pbar.set_description("Epoch %3d - Loss %8.5f " % (epoch, loss_mean))

    # confirm output
    imgshow(args.image_output_dir + "real" + str(epoch),
            make_grid(data_input.view(-1, 1, image_size, image_size).data))
    imgshow(args.image_output_dir + "generate" + str(epoch),
            make_grid(new_visible.view(-1, 1, image_size, image_size).data))
    imgshow(args.image_output_dir + "parameter" + str(epoch),
            make_grid(rbm.W.view(hidden_layers, 1, image_size, image_size).data))
    imgshow(args.image_output_dir + "hidden" + str(epoch),
            make_grid(hidden.view(-1, 1, args.hidden_size, args.hidden_size).data))

    np.savetxt(args.text_output_dir + "W.data" +
               str(epoch), rbm.W.data.numpy())
    # .data is used to retrieve the tensor held by the Parameter(Variable) W, then we can get the numpy representation


# Save the model
torch.save(rbm.state_dict(), "trained_rbm.pytorch")
loss_file.close()
