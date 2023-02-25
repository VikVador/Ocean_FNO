#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# -------
# Credits
# -------
# For the source code, all the credit goes to:
#
#       A. Ross, Z. Li, P. Perezhogin, C. Fernandez-Granda & L. Zanna
#
# with the code originally coming from their paper:
#
#       https://www.essoar.org/pdfjs/10.1002/essoar.10511742.1
#

import os
import glob
import time
import pyqg
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr

from collections                              import OrderedDict
from torch.autograd                           import grad, Variable
from pyqg_parameterization_benchmarks.utils   import FeatureExtractor, Parameterization

#----------------------------------------------------------------------------------------
#                 Convolutional neural network - Architecture definition
#----------------------------------------------------------------------------------------
# If you want to modify the core of the FCNN, i.e. it's structure, then it's here !
class FullyCNN(nn.Sequential):
    """
    Pytorch class defining our CNN architecture, plus some helpers for  dealing with constraints and scaling.
    """
    def __init__(self, inputs, targets, padding = 'circular', zero_mean = True):

        # Defining the padding type
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding in ['same', 'circular']:
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')

        # Dimension of input and output data
        n_in = len(inputs)
        n_out = len(targets)

        # Storing more information about the data and parameters
        self.padding      = padding
        self.inputs       = inputs
        self.targets      = targets
        self.is_zero_mean = zero_mean
        self.n_in         = n_in
        kw = {}
        if padding == 'circular':
            kw['padding_mode'] = 'circular'

    #-----------------------------------------------------------------------------------
    #                    MODIFY - FULLY CONVOLUTIONAL NEURAL NETWORK
    #-----------------------------------------------------------------------------------
        block1 = self._make_subblock(nn.Conv2d(n_in, 128, 5, padding = padding_5, **kw))
        block2 = self._make_subblock(nn.Conv2d(128,   64, 5, padding = padding_5, **kw))
        block3 = self._make_subblock(nn.Conv2d(64,    32, 3, padding = padding_3, **kw))
        block4 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        block5 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        block6 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        block7 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        conv8  =                     nn.Conv2d(32, n_out, 3, padding = padding_3)

        # Combining everything together
        super().__init__(*block1, *block2, *block3, *block4, *block5, *block6, *block7, conv8)

    def _make_subblock(self, conv):
        return [conv, nn.ReLU(), nn.BatchNorm2d(conv.out_channels)]

    #-----------------------------------------------------------------------------------
    #                    MODIFY - FULLY CONVOLUTIONAL NEURAL NETWORK
    #-----------------------------------------------------------------------------------
    # Compute a forward pass of the FCNN using the input X
    def forward(self, x):
        r = super().forward(x)
        if self.is_zero_mean:
            return r - r.mean(dim = (1,2,3), keepdim = True)
        else:
            return r

    # Used to extract and write (human-readable) the features (I think)
    def extract_vars(self, m, features, dtype=np.float32):
        ex  = FeatureExtractor(m)
        arr = np.stack([np.take(ex(feat), z, axis=-3) for feat, z in features], axis=-3)
        arr = arr.reshape((-1, len(features), ex.nx, ex.nx))
        arr = arr.astype(dtype)
        return arr

    # From dataset m, the variables in inputs are extracted (ex: inputs = ['q', 'u'])
    def extract_inputs(self, m):
        return self.extract_vars(m, self.inputs)

    # From dataset m, the variables in targets are extracted (ex: inputs = ['q_subgrid_forcing'])
    def extract_targets(self, m):
        return self.extract_vars(m, self.targets)

    # As weird as it sounds, it computes and stores all the gradients associated to each minibatch
    def input_gradients(self, inputs, output_channel, j, i, device=None):
        if device is None:
            print(torch.cuda.is_available())
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)

        X = self.input_scale.transform(self.extract_inputs(inputs))

        grads = []
        for x_, in minibatch(X, shuffle=False, as_tensor=False):
            x = Variable(torch.tensor(x_), requires_grad=True).to(device)
            y = self.forward(x)[:,output_channel,j,i]
            grads.append(grad(y.sum(), x)[0].cpu().numpy())

        grads = self.output_scale.inverse_transform(np.vstack(grads))

        s = list(inputs.q.shape)
        grads = np.stack([
            grads[:,i].reshape(s[:-3] + s[-2:])
            for i in range(len(self.targets))
        ], axis=-3)

        if isinstance(inputs, pyqg.Model):
            return grads.astype(inputs.q.dtype)
        else:
            return grads

    # Used to compute a prediction (for testing not training since grad not saved)
    def predict(self, inputs, device = None):

        # Checking for a GPU !
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)

        # Loading the input and normalizing them !
        X = self.input_scale.transform(self.extract_inputs(inputs))

        # Compute predictions
        preds = []
        for x, in minibatch(X, shuffle = False):
            x = x.to(device)
            with torch.no_grad():
                preds.append(self.forward(x).cpu().numpy())

        # Loading the outputs and normalizing them !
        preds = self.output_scale.inverse_transform(np.vstack(preds))

        s     = list(inputs.q.shape)
        preds = np.stack([preds[:,i].reshape(s[:-3] + s[-2:]) for i in range(len(self.targets))], axis = -3)

        try:
            return preds.astype(inputs.q.dtype)
        except:
            return preds

    # Defining own version of the mean squarred error
    def mse(self, inputs, targets, **kw):
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true)**2, axis=1))

    # Used to start training a model
    def fit(self, inputs, targets, rescale = False, **kw):

        # Creation of a scaler for the input (mu = 0 and std = 1) if:
        # - Rescale = true
        # - The self variable 'input_scale' is not found
        # - The self variable 'input_scale' is found BUT equal to NONE
        if rescale or not hasattr(self, 'input_scale') or self.input_scale is None:
            self.input_scale = ChannelwiseScaler(inputs)

        # Creation of a scaler for the output (mu = 0 and std = 1) if:
        # - Rescale = true
        # - The self variable 'input_scale' is not found
        # - The self variable 'input_scale' is found BUT equal to NONE
        if rescale or not hasattr(self, 'output_scale') or self.output_scale is None:
            self.output_scale = ChannelwiseScaler(targets, zero_mean=self.is_zero_mean)

        # Going further into the training
        train(self, self.input_scale.transform(inputs), self.output_scale.transform(targets), **kw)

    # Save the model !
    def save(self, path):

        # Creation of a new directory using terminal command
        os.system(f"mkdir -p {path}")

        # Defining device to use (GPU or CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Saving the models' weights
        self.cpu()
        torch.save(self.state_dict(), f"{path}/weights.pt")

        # Saving data and other parameters
        self.to(device)
        if hasattr(self, 'input_scale') and self.input_scale is not None:
            with open(f"{path}/input_scale.pkl", 'wb') as f:
                pickle.dump(self.input_scale, f)

        if hasattr(self, 'output_scale')  and self.output_scale is not None:
            with open(f"{path}/output_scale.pkl", 'wb') as f:
                pickle.dump(self.output_scale, f)

        with open(f"{path}/inputs.pkl", 'wb') as f:
            pickle.dump(self.inputs, f)

        with open(f"{path}/targets.pkl", 'wb') as f:
            pickle.dump(self.targets, f)

        if self.is_zero_mean:
            open(f"{path}/zero_mean", 'a').close()

        if hasattr(self, 'padding'):
            with open(f"{path}/padding", 'w') as f:
                f.write(self.padding)

    # Used to load a FCNN that has been saved
    @classmethod
    def load(cls, path, set_eval = True, **kwargs):

        # Loading the associated inputs and targets used for training
        with open(f"{path}/inputs.pkl", 'rb') as f:
            inputs = pickle.load(f)
        with open(f"{path}/targets.pkl", 'rb') as f:
            targets = pickle.load(f)

        # Loading the padding value
        kw = {}
        kw.update(**kwargs)
        if os.path.exists(f"{path}/padding"):
            with open(f"{path}/padding", 'r') as f:
                kw['padding'] = f.read().strip()

        # Initialization of the model
        model = cls(inputs, targets, **kw)

        # Inserting former parameters
        model.load_state_dict(torch.load(f"{path}/weights.pt"))

        # Loading other parameters
        if os.path.exists(f"{path}/input_scale.pkl"):
            with open(f"{path}/input_scale.pkl", 'rb') as f:
                model.input_scale = pickle.load(f)
        if os.path.exists(f"{path}/output_scale.pkl"):
            with open(f"{path}/output_scale.pkl", 'rb') as f:
                model.output_scale = pickle.load(f)
        if os.path.exists(f"{path}/zero_mean"):
            model.is_zero_mean = True

        # By default, model in evalutation model (important if dropout or other stuff)
        if set_eval:
            model.eval()

        return model

#----------------------------------------------------------------------------------------
#                                 Statistics functions
#----------------------------------------------------------------------------------------
class BasicScaler(object):
    """
    Simple class to perform normalization and denormalization
    """
    def __init__(self, mu = 0, sd = 1):
        self.mu = mu
        self.sd = sd

    def transform(self, x):
        return (x - self.mu) / self.sd

    def inverse_transform(self, z):
        # One needs to compute first mu and sd -> ChannelwiseScaler
        return z * self.sd + self.mu

class ChannelwiseScaler(BasicScaler):
    """
    Simple class to compute mean and standard deviation of each channel
    """
    def __init__(self, x, zero_mean = False):
        assert len(x.shape) == 4

        # Computation of the mean
        if zero_mean:
            mu = 0
        else:
            mu = np.array([x[:,i].mean() for i in range(x.shape[1])])[np.newaxis , : , np.newaxis , np.newaxis]

        # Computation of the standard deviation
        sd = np.array([x[:,i].std() for i in range(x.shape[1])])[np.newaxis , : , np.newaxis , np.newaxis]

        # Initialization of a scaler with correct mean and standard deviation
        super().__init__(mu, sd)

#----------------------------------------------------------------------------------------
#                 Convolutional neural network - Information over terminal
#----------------------------------------------------------------------------------------
# Used to print a basic section title in terminal
def section(title = "UNKNOWN"):

    # Number of letters to determine section size
    title_size = len(title)

    # Section title boundaries
    boundary  = "-"
    for i in range(title_size + 1):
        boundary += "-"

    # Printing section
    print(boundary)
    print(f" {title} ")
    print(boundary)

# Used to display a simple progress bar while training for 1 epoch
def progressBar(loss_training, estimated_time_epoch, nb_epoch_left, percent, width = 40):

    # Setting up the useful information
    left          = width * percent // 100
    right         = width - left
    tags          = "#" * int(left)
    spaces        = " " * int(right)
    percents      = f"{percent:.2f} %"
    loss_training = f"{loss_training * 1:.4f}"

    # Computing timings
    estimated_time_total = f"{nb_epoch_left * estimated_time_epoch:.2f} s"

    # Displaying a really cool progress bar !
    print("\r[", tags, spaces, "] - ", percents, " | Loss (Training) = ", loss_training,
          " | Total time left : ", estimated_time_total, " | ", sep = "", end = "", flush = True)

#----------------------------------------------------------------------------------------
#             MODIFY - Convolutional neural network - Training functions
#----------------------------------------------------------------------------------------
def minibatch(*arrays, batch_size = 64, as_tensor = True, shuffle = True):

    # Since set removes duplicate, this assert make sure that inputs and outputs have same dimensions !
    assert len(set([len(a) for a in arrays])) == 1

    # Index vector
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)

    # Step size (arrondis vers le bas)
    steps = int(np.ceil(len(arrays[0]) / batch_size))

    # Choose data type to store the batch
    xform = torch.as_tensor if as_tensor else lambda x: x

    # Creation of all the mini batches !
    for step in range(steps):
        idx = order[step * batch_size : (step + 1) * batch_size]

        # Yield is the same as return except that it return a generator ! In other words,
        # it is an iterator that you can only go through once since values are discarded !
        # This really really really smart !
        yield tuple(xform(array[idx]) for array in arrays)

# Used to make the training a model, it could be nice to upgrade it !
def train(net, inputs, targets, num_epochs = 50, batch_size = 64, learning_rate = 0.001, device = None):

    # Looking for a GPU !
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

    # Training parameters
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs/2), int(num_epochs * 3/4), int(num_epochs * 7/8)], gamma=0.1)
    criterion = nn.MSELoss()

    # Used to compute training progression bar (1)
    size_train = len(inputs)
    epoch_time = 0

    # Start training
    for epoch in range(num_epochs):
        epoch_loss  = 0.0
        epoch_steps = 0

        # Display useful information over terminal (1)
        print("Epoch : ", epoch + 1, "/", num_epochs)

        # Used to compute training progression bar (2)
        index = batch_size

        # Used to approximate time left for current epoch and in total
        start      = time.time()

        # Retreiving a batch of data
        for x, y in minibatch(inputs, targets, batch_size = batch_size):

            # Reseting gradients
            optimizer.zero_grad()

            # Computing prediction and storing target
            yhat  = net.forward(x.to(device))
            ytrue = y.to(device)

            # Computing loss
            loss = criterion(yhat, ytrue)
            loss.backward()

            # Optimization
            optimizer.step()

            # Updating epoch info ! Would be nice to upgrade it !
            epoch_loss   += loss.item()
            epoch_steps  += 1
            nb_epoch_left = num_epochs - epoch
            percentage    = (index/size_train) * 100 if (index/size_train) <= 1 else 100

            # Displaying information over terminal (2)
            progressBar(epoch_loss/epoch_steps, epoch_time, nb_epoch_left, percentage)
            index += batch_size

        # Updating timing
        epoch_time    = time.time() - start

        # Just to make sure there is no overlap between progress bar and section
        print(" ")

        # Updating the scheduler to update learning rate !
        scheduler.step()

#----------------------------------------------------------------------------------------
#                      Convolutional neural network - Handler
#----------------------------------------------------------------------------------------
# Contains all the functions one needs to train the NN !
class FCNNParameterization(Parameterization):

    # Associates a/some model/s to the handler, ATTENTION model should be an instance of FullyCNN
    def __init__(self, directory, models = None, **kw):
        self.directory = directory
        self.models = models if models is not None else [
            FullyCNN.load(f, **kw)
            for f in sorted(glob.glob(os.path.join(directory, "models/*")))
        ]

    # Retreives all the targets associated to each model
    @property
    def targets(self):
        targets = set()
        for model in self.models:
            for target, z in model.targets:
                targets.add(target)
        return list(sorted(list(targets)))

    # Compute prediction for all models
    def predict(self, m):
        preds = {}

        for model in self.models:
            pred = model.predict(m)
            assert len(pred.shape) == len(m.q.shape)

            # ------ DON'T WORRY ------
            # Handle the arduous task of getting the indices right for many possible input shapes
            # (e.g. pyqg.Model or xr.Dataset snapshot stack)
            for channel in range(pred.shape[-3]):
                target, z = model.targets[channel]
                if target not in preds:
                    preds[target] = np.zeros_like(m.q)
                out_indices     = [slice(None) for _ in m.q.shape]
                out_indices[-3] = slice(z,z+1)
                in_indices      = [slice(None) for _ in m.q.shape]
                in_indices[-3]  = slice(channel,channel+1)
                preds[target][tuple(out_indices)] = pred[tuple(in_indices)]
            # ------ DON'T WORRY ------

        return preds

    @classmethod
    def train_on(cls, dataset, directory, inputs = ['q', 'u', 'v'], targets = ['q_subgrid_forcing'],
                      num_epochs = 50, zero_mean = True, padding = 'circular', **kw):

        # Retreives the numbzr of layers of the simulation
        layers = range(len(dataset.lev))

        # For each layer, one creates a FCNN which has:
        # - Inputs    : 'q', 'u', 'v' of each layer
        # - Outputs   : 'q_subgrid_forcing' of the current layer
        # - Zero mean : true
        # - Padding   : 'circular'
        models = [
            FullyCNN(
                [(feat, zi) for feat in inputs for zi in layers], # For each layer, one gives as input 'q', 'u', 'v'
                [(feat, z) for feat in targets],                  # For each layer, one computes
                zero_mean = zero_mean,
                padding = padding
            ) for z in layers
        ]

        # Displaying information over terminal (1)
        section("Fully Convolutional neural network - Training")

        # Stores the trained models
        trained = []

        # Looping over each model
        for z, model in enumerate(models):

            # Creation of the model path
            model_dir = os.path.join(directory, f"models/{z}")

            # Displaying information over terminal (2)
            section(f"Model - z = {z}")

            # Already exists (What is model 2)
            if os.path.exists(model_dir):
                trained.append(FullyCNN.load(model_dir))

            # Training of the current model
            else:

                # Retreives the input and outputs
                X = model.extract_inputs(dataset)
                Y = model.extract_targets(dataset)

                # Training the model
                model.fit(X, Y, num_epochs = num_epochs, **kw)

                # Saving the model
                model.save(os.path.join(directory, f"models/{z}"))

                # Adding model to the list of trained one
                trained.append(model)

        # Return the trained parameterization
        return cls(directory, models = trained)
