#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# -----------------
#     Librairies
# -----------------
#
# --------- Standard ---------
import os
import glob
import time
import pyqg
import torch
import pickle
import numpy          as np
import xarray         as xr
import torch.nn       as nn
import torch.optim    as optim
from   torch.autograd import grad, Variable

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils_TFE  import *
from pyqg_parameterization_benchmarks.plots_TFE  import *
from pyqg_parameterization_benchmarks.utils      import FeatureExtractor, Parameterization

# ----------------------------------------------------------------------------------------------------------
#
#
#                            Parameterization - Fully convolutionnal Neural Network
#
#
# ----------------------------------------------------------------------------------------------------------
class FullyCNN(nn.Sequential):

    def __init__(self, inputs, targets, padding = 'circular', zero_mean = True):

        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding in ['same', 'circular']:
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('FullyCNN - Unknow value for padding parameter, i.e. should be None or circular')

        # Dimension of input and output data
        n_in  = len(inputs)
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
        #                                   Architecture
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
    #                                     Forward
    #-----------------------------------------------------------------------------------
    def forward(self, x):
        
        # Final prediction
        pred = super().forward(x)
        
        # Normalization of prediction
        if self.is_zero_mean:
            return pred - pred.mean(dim = (1,2,3), keepdim = True)
        else:
            return pred

    # ------------------------------------------------------------------------------------------------------
    #                                            Extract information
    # ------------------------------------------------------------------------------------------------------
    # Used to extract and write (human-readable) the features
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

    # ------------------------------------------------------------------------------------------------------
    #                             Predictions (for testing not training since grad not saved)
    # ------------------------------------------------------------------------------------------------------
    def predict(self, inputs, device = None):

        # Checking for a GPU !
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)

        # Loading the input and normalizing them !
        X = self.input_scale.transform(self.extract_inputs(inputs))

        # Compute predictions
        preds = []
        
        # Retreive inputs
        for x, in minibatch(X, shuffle = False):
            
            # Sending to GPU
            x = x.to(device)
            
            # Computing predictions
            with torch.no_grad():
                preds.append(self.forward(x).cpu().numpy())

        # Concatenation and normalization of outputs
        preds = self.output_scale.inverse_transform(np.vstack(preds))

        # Shaping predictions
        s     = list(inputs.q.shape)
        preds = np.stack([preds[:,i].reshape(s[:-3] + s[-2:]) for i in range(len(self.targets))], axis = -3)

        try:
            return preds.astype(inputs.q.dtype)
        except:
            return preds

    # ------------------------------------------------------------------------------------------------------
    #                             Fitting (Prepares inputs/outputs and train the model)
    # ------------------------------------------------------------------------------------------------------
    # Used to start training a model
    def fit_FCNN(self, inputs, targets, inputs_validation, targets_validation, 
                 level, tensorboard, rescale = False, **kw):

        # -------------- Scaler creation --------------
        if rescale or not hasattr(self, 'input_scale') or self.input_scale is None:
            self.input_scale = ChannelwiseScaler(inputs)
            inputs_val_scale = ChannelwiseScaler(inputs_validation)

        if rescale or not hasattr(self, 'output_scale') or self.output_scale is None:
            self.output_scale = ChannelwiseScaler(targets,            zero_mean = self.is_zero_mean)
            outputs_val_scale = ChannelwiseScaler(targets_validation, zero_mean = self.is_zero_mean)

        # -------------- Scaling --------------
        scaled_inputs             = self.input_scale.transform(inputs)
        scaled_targets            = self.output_scale.transform(targets)
        scaled_inputs_validation  = inputs_val_scale.transform(inputs_validation)
        scaled_targets_validation = outputs_val_scale.transform(targets_validation)
        
        # ---------------- Training ----------------
        train_FCNN(self, scaled_inputs, scaled_targets, scaled_inputs_validation, scaled_targets_validation, \
                   level, tensorboard = tensorboard, **kw)

    # ------------------------------------------------------------------------------------------------------
    #                                          Other class functions
    # ------------------------------------------------------------------------------------------------------
    # Defining own version of the mean squarred error
    def mse(self, inputs, targets, **kw):
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true)**2, axis=1))
    
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

# ----------------------------------------------------------------------------------------------------------
#
#                                         Parameterization - FCNN Training
#
# ----------------------------------------------------------------------------------------------------------
def train_FCNN(model, inputs, targets, inputs_validation, targets_validation, level, tensorboard, 
               num_epochs = 50, batch_size = 64, learning_rate = 0.001, device = None):

    # Looking for a GPU !
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # Training properties
    mls_list  = [int(num_epochs/2), int(num_epochs * 3/4), int(num_epochs * 7/8)]
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = mls_list, gamma = 0.1)
    criterion = nn.MSELoss()

    # Used to compute training progression bar (1)
    size_train = len(inputs)
    epoch_time = 0
    
    # Custom layout for tensorboard vizualization
    layout = {
        f"Level : {str(level)}": {
            "Training"    : ["Multiline", ["loss/train"]],
            "Validation"  : ["Multiline", ["loss/validation"]],
        },
    }
    
    # Adding layout
    tensorboard.add_custom_scalars(layout)

    # ------------------------------------------------------------------------------------------------------
    #                                                 Fitting
    # ------------------------------------------------------------------------------------------------------
    for epoch in range(num_epochs):
        
        # Initialization
        epoch_loss      = 0.0
        epoch_loss_val  = 0.0
        epoch_steps     = 0
        epoch_steps_val = 0

        # Display useful information over terminal (1)
        print("Epoch : ", epoch + 1, "/", num_epochs)

        # Used to compute training progression bar (2)
        index = batch_size

        # Used to approximate time left for current epoch and in total
        start      = time.time()

        # --------------------------------------------------------------------------------------------------
        #                                             Training
        # --------------------------------------------------------------------------------------------------
        for x, y in minibatch(inputs, targets, batch_size = batch_size):

            # Vizualizing model on tensorboard
            if epoch == 0 and index == batch_size:
                tensorboard.add_graph(model, input_to_model = x.to(device), verbose = False)
            
            # Reseting gradients
            optimizer.zero_grad()

            # Computing prediction and storing target
            yhat  = model.forward(x.to(device))
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
            progressBar(epoch_loss/epoch_steps, 0, epoch_time, nb_epoch_left, percentage)
            index += batch_size
            
        # Updating tensorboard (1)
        tensorboard.add_scalar("loss/train", epoch_loss/epoch_steps, global_step = epoch)
            
        # --------------------------------------------------------------------------------------------------
        #                                             Validation
        # --------------------------------------------------------------------------------------------------     
        with torch.no_grad():  
            
            # Retreiving a batch of data
            for x, y in minibatch(inputs_validation, targets_validation, batch_size = batch_size):

                # Computing prediction and storing target
                yhat  = model.forward(x.to(device))
                ytrue = y.to(device)

                # Computing loss
                loss = criterion(yhat, ytrue)

                # Updating epoch info ! Would be nice to upgrade it !
                epoch_loss_val  += loss.item()
                epoch_steps_val += 1
                
                # Displaying information over terminal (2)
                progressBar(epoch_loss/epoch_steps, epoch_loss_val/epoch_steps_val, epoch_time, nb_epoch_left, percentage)
                
            # Updating tensorboard (2)
            tensorboard.add_scalar("loss/validation", epoch_loss_val/epoch_steps_val, global_step = epoch)
                
        # Updating timing
        epoch_time    = time.time() - start

        # Just to make sure there is no overlap between progress bar and section
        print(" ")

        # Updating the scheduler to update learning rate !
        scheduler.step()

# ----------------------------------------------------------------------------------------------------------
#
#                                         Parameterization - FCNN handler
#
# ----------------------------------------------------------------------------------------------------------
class FCNNParameterization(Parameterization):

    def __init__(self, directory, models = None, **kw):
        self.directory = directory
        self.models    = models if models is not None else [
                         FullyCNN.load(f, **kw)
                         for f in sorted(glob.glob(os.path.join(directory, "models/*")))]

    @property
    def targets(self):
        """
        Retreive all the targets associated to each model
        """
        targets = set()
        for model in self.models:
            for target, z in model.targets:
                targets.add(target)
        return list(sorted(list(targets)))

    # --------------------------------------------------------------------------------------------------
    #                                         Predict (Handler)
    # --------------------------------------------------------------------------------------------------  
    def predict(self, m):
        
        # Stores predictions
        preds = {}

        # Prediction of each model (one per layer)
        for model in self.models:
            
            # Compute predictions
            pred = model.predict(m)
            
            # Security
            assert len(pred.shape) == len(m.q.shape)

            # Handle the arduous task of getting the indices right for many possible input shapes (e.g. pyqg.Model or xr.Dataset snapshot stack)
            for channel in range(pred.shape[-3]):
                target, z = model.targets[channel]
                if target not in preds:
                    preds[target] = np.zeros_like(m.q)
                out_indices       = [slice(None) for _ in m.q.shape]
                out_indices[-3]   = slice(z,z+1)
                in_indices        = [slice(None) for _ in m.q.shape]
                in_indices[-3]    = slice(channel,channel+1)
                preds[target][tuple(out_indices)] = pred[tuple(in_indices)]
        return preds

    # --------------------------------------------------------------------------------------------------
    #                                         Training (Handler)
    # --------------------------------------------------------------------------------------------------   
    @classmethod
    def train_on(cls, dataset, dataset_validation, directory, tensorboard, 
                 inputs = ['q', 'u', 'v'], targets = ['q_subgrid_forcing'], 
                 num_epochs = 50, zero_mean = True, padding = 'circular', **kw):

        # Retreives the numbzr of layers of the simulation
        layers = range(len(dataset.lev))

        models = [
            FullyCNN(
                [(feat, zi) for feat in inputs for zi in layers], 
                [(feat, z) for feat in targets],                  
                zero_mean = zero_mean,
                padding = padding
            ) for z in layers
        ]

        # Displaying information over terminal (1)
        section("Parameterization - Training")

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

                # Retreives the input and outputs of training set
                X = model.extract_inputs(dataset)
                Y = model.extract_targets(dataset)
                
                # Retreives the input and outputs of validation set
                X_val = model.extract_inputs(dataset_validation)
                Y_val = model.extract_targets(dataset_validation)

                # Training the model
                model.fit_FCNN(X, Y, X_val, Y_val, z, tensorboard, num_epochs = num_epochs, **kw)

                # Saving the model
                model.save(os.path.join(directory, f"models/{z}"))

                # Adding model to the list of trained one
                trained.append(model)

        # Return the trained parameterization
        return cls(directory, models = trained)
