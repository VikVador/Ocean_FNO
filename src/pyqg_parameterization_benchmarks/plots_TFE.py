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
#   Documentation
# -----------------
# This file contains all the functions used to make graphics throughout all the .py
# (terminal, notebooks plots, saved plots, ...)
#
# -----------------
#     Librairies
# -----------------
#
# --------- Standard ---------
import os
import numpy             as np
import pandas            as pd
import xarray            as xr
import seaborn           as sns
import matplotlib.pyplot as plt

# --------- PYQG ---------
import pyqg
from   pyqg.diagnostic_tools import calc_ispec         as _calc_ispec

calc_ispec = lambda *args, **kwargs: _calc_ispec(*args, averaging = False, truncate =False, **kwargs)

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils           import *
from pyqg_parameterization_benchmarks.utils_TFE       import *

# ----------------------------------------------------------------------------------------------------------
#
#                                                    Terminal
#
# ----------------------------------------------------------------------------------------------------------
def tfe_title():
    """
    Documentation
    -------------
    Display the title of the TFE (used for esthetics and make sure the script is running)
    """
    print("------------------------------------------------------------------------------")
    print("                                                                              ")
    print("           Ocean subgrid parameterization using machine learning              ")
    print("                                                                              ")
    print("                             Graduation work                                  ")
    print("                                                                              ")
    print("------------------------------------------------------------------------------")
    print("@ Victor Mangeleer\n")

def section(title = "UNKNOWN"):
    """
    Documentation
    -------------
    Used to print a basic section title in terminal
    """
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
    
def show_sim_parameters(parameters):
    """
    Documentation
    -------------
    Used to display important variables used by generate_dataset.py
    """
    print("Simulation type      = " + str(parameters.simulation_type))
    print("Target sample size   = " + str(parameters.target_sample_size))
    print("Operator             = " + str(parameters.operator_cf))
    print("Number of threads    = " + str(parameters.nb_threads))
    print("Memory               = " + str(parameters.memory))
    print("Save folder          = " + parameters.save_folder)
    print("Save high resolution = " + str(parameters.save_high_res))
    print("Skipped time         = " + str(parameters.skipped_time))
    
def show_param_parameters(parameters):
    """
    Documentation
    -------------
    Used to display important variables used by train_parameterization.py
    """
    print("Folder training = ")
    for f in parameters.folder_training:
        print("                      - " + f)
    print("\nFolder validation = ")
    for f in parameters.folder_validation:
        print("                      - " + f)
    print("\nInputs            = ")
    for f in parameters.inputs:
        print("                      - " + f)
    print("\nParam. type       = " + parameters.param_type)
    print("Param. name       = " + parameters.param_name)
    print("Targets           = " + parameters.targets)
    print("Number of epochs  = " + str(parameters.num_epochs))
    print("Zero mean         = " + str(parameters.zero_mean))
    print("Padding           = " + parameters.padding)
    print("Memory            = " + str(parameters.memory) + "\n")
    
def show_models_offline(root, datasets, models):
    """
    Documentation
    -------------
    Used to display important variables used by offline.py
    """
    print("Folder offline      = ")
    for d in datasets:
        print("                      - " + d)
    print("\nModels              = ")
    for m in models:
        print("                      - " + m)
    print("\nModel folder (root) = " + root)

# ----------------------------------------------------------------------------------------------------------
#
#                                                    Training
#
# ----------------------------------------------------------------------------------------------------------
# Used to display a simple progress bar while training for 1 epoch
def progressBar(loss_training, loss_validation, estimated_time_epoch, nb_epoch_left, percent, width = 40):

    # Setting up the useful information
    left            = width * percent // 100
    right           = width - left
    tags            = "#" * int(left)
    spaces          = " " * int(right)
    percents        = f"{percent:.2f} %"
    loss_training   = f"{loss_training * 1:.3f}"
    loss_validation = f"{loss_validation * 1:.3f}"

    # Computing timings
    estimated_time_total = f"{nb_epoch_left * estimated_time_epoch:.2f} s"

    # Displaying a really cool progress bar !
    print("\r[", tags, spaces, "] - ", percents, " | Loss (Training) = ", loss_training, " | Loss (Validation) = ", loss_validation,
          " | Total time left : ", estimated_time_total, " | ", sep = "", end = "", flush = True)
    
# ----------------------------------------------------------------------------------------------------------
#
#                                                  Offline testing
#
# ----------------------------------------------------------------------------------------------------------
def colorbar(label):
    """
    Documentation
    -------------
    Used to make a good ass looking color bar for plots
    """
    plt.colorbar().set_label(label, fontsize = 16, rotation = 0, ha = 'left', va = 'center')
    
def comparaison_plot_offline(metrics, model_folder, x = "Model", y = 'R2_z0', ylab = "$R^2$ ($Z$ = 0)", min_v = 0, max_v = 1):
    """
    Documentation
    -------------
    Used to make a comparaison of the metrics (R^2 and correlation) in a bar plot fashion
    """
    plot = sns.barplot(x = x, y = y, data = metrics.sort_values(y), palette = 'magma')
    plot.set(ylim = (min_v - 0.02, max_v + 0.02), xlabel = None, ylabel = ylab)
    plt.xticks(rotation = 90, fontsize = 12)
    plot.figure.savefig(model_folder + f"___PLOTS___/{y}.svg", bbox_inches = "tight")
    plt.show()
    plt.close()
    
def beautify_model_names(model_names):
    
    # Stores all the model names that have been beautify
    beauty_model_names = []
    
    for n in model_names:
        
        # Currently beautified name
        beauty = ""
        
        # Base
        if "BASE" in n:
            beauty_model_names.append("BASE_q → $S_{q_{total}, 5000}$")
            continue
            
        # Parameterization name
        if "FCNN"   in n:
            beauty += "FCNN_"
        if "KASKADE" in n:
            beauty += "KASKADE_"
            
        # Input beautification
        if 'q_to' in n:
            beauty += "q → "
        if "q_u_to" in n:
            beauty += "(q, u) → "
        if "q_v_to" in n:
            beauty += "(q, v) → "
        if "q_u_v_to" in n:
            beauty += "(q, u, v) → "    
    
        # Training information beautification
        nb_samples = ""
        if "500" in n:
            nb_samples = ",500"
        if "1000" in n:
            nb_samples = ",1000"
        if "2000" in n:
            nb_samples = ",2000"
        if "3000" in n:
            nb_samples = ",3000"
        if "4000" in n:
            nb_samples = ",4000"
        if "5000" in n:
            nb_samples = ",5000"
        if "6000" in n:
            nb_samples = ",5000"
        if "7000" in n:
            nb_samples = ",5000"
        if "8000" in n:
            nb_samples = ",5000"
        if "9000" in n:
            nb_samples = ",5000"
        if "10000" in n:
            nb_samples = ",10000"
        if "20000" in n:
            nb_samples = ",20000"  
            
        # Output beautification
        if "q_subgrid_forcing" in n:
            beauty += "$S_{q" + nb_samples + "}$"    
        if "q_forcing_total"   in n:
            beauty += "$S_{q_{total}" + nb_samples + "}$"
        if "uq_subgrid_flux"   in n:
            beauty += "$\Phi_{(u,q)" + nb_samples + "}$"
        if "vq_subgrid_flux"   in n:
            beauty += "$\Phi_{(v,q)" + nb_samples + "}$"
            
        beauty_model_names.append(beauty)
            
    return beauty_model_names
        
# ----------------------------------------------------------------------------------------------------------
#
#                                                  State variables
#
# ----------------------------------------------------------------------------------------------------------
# Allows to display easily an image (from coarsening notebook)
def imshow(arr, vlim = 3e-5):
    plt.xticks([]); plt.yticks([])
    return plt.imshow(arr, vmin = -vlim, vmax = vlim, cmap = 'bwr', interpolation = 'none')

# Used to plot easily a state variable
def plotStateVariable(high_res, low_res, state_variable = "q", save_path = ""):

    # Text for the caption
    caption = ["Upper Level : z = 1", "Lower Level : z = 2"]

    # Looping over the levels
    for l in range(2):

        # Initialization of the plot
        fig = plt.figure(figsize=(21, 6))

        # Plotting the state variables (Note: in the coarsening class, the original high resolution model is stored in m1 !)
        if state_variable == "q":

            # High resolution
            plt.subplot(1, 2, 1)
            high_res.q.isel(lev = l, time = -1).plot()

            # Low resolution
            plt.subplot(1, 2, 2)
            low_res.q.isel(lev = l, time = -1).plot()

            # Adding a caption to the plot
            fig.text(0.45, -0.1, f"$Figure$: Representation of the potential vorticity q for the high resolution (left) and low resolution simulations (right) - {caption[l]}", ha = 'center')

        elif state_variable == "u":

            # High resolution
            plt.subplot(1, 2, 1)
            high_res.u.isel(lev = l, time = -1).plot()

            # Low resolution
            plt.subplot(1, 2, 2)
            low_res.u.isel(lev = l, time = -1).plot()

            # Adding a caption to the plot
            fig.text(0.45, -0.1, f"$Figure$: Representation of the horizontal velocity u for the high resolution (left) and low resolution simulations (right) - {caption[l]}", ha = 'center')

        elif state_variable == "v":
                    
            # High resolution
            plt.subplot(1, 2, 1)
            high_res.v.isel(lev = l, time = -1).plot()

            # Low resolution
            plt.subplot(1, 2, 2)
            low_res.v.isel(lev = l, time = -1).plot()

            # Adding a caption to the plot
            fig.text(0.45, -0.1, f"$Figure$: Representation of the vertical velocity y for the high resolution (left) and low resolution simulations (right) - {caption[l]}", ha = 'center')

        elif state_variable == "ufull":
                    
            # High resolution
            plt.subplot(1, 2, 1)
            high_res.ufull.isel(lev = l, time = -1).plot()

            # Low resolution
            plt.subplot(1, 2, 2)
            low_res.ufull.isel(lev = l, time = -1).plot()

            # Adding a caption to the plot
            fig.text(0.45, -0.1, f"$Figure$: Representation of the horizontal velocity u with background flow for the high resolution (left) and low resolution simulations (right) - {caption[l]}", ha = 'center')


        elif state_variable == "vfull":
                    
            # High resolution
            plt.subplot(1, 2, 1)
            high_res.vfull.isel(lev = l, time = -1).plot()

            # Low resolution
            plt.subplot(1, 2, 2)
            low_res.vfull.isel(lev = l, time = -1).plot()

            # Adding a caption to the plot
            fig.text(0.45, -0.1, f"$Figure$: Representation of the vertical velocity v with background flow for the high resolution (left) and low resolution simulations (right) - {caption[l]}", ha = 'center')


        elif state_variable == "streamfunction":
                    
            # High resolution
            plt.subplot(1, 2, 1)
            high_res.streamfunction.isel(lev = l, time = -1).plot()

            # Low resolution
            plt.subplot(1, 2, 2)
            low_res.streamfunction.isel(lev = l, time = -1).plot()

            # Adding a caption to the plot
            fig.text(0.45, -0.1, f"$Figure$: Representation of the streamfunction for the high resolution (left) and low resolution simulations (right) - {caption[l]}", ha = 'center')

        # Saving results
        save_path_complete = f"{save_path}/state_variables/{state_variable}"

        # Check if image folder exists
        if not os.path.exists(save_path_complete):
            os.makedirs(save_path_complete)

        # Save the figure
        fig.savefig(save_path_complete + f"/{state_variable}_{l}.png", bbox_inches = "tight")

# ----------------------------------------------------------------------------------------------------------
#
#                                               Subgrid variables
#
# ----------------------------------------------------------------------------------------------------------
# Allows to display easily an image (from coarsening notebook, version 2 with axis label)
def imshow_2(arr, vlim = 3e-5):
    plt.xticks([]); plt.yticks([])
    plt.xlabel("Grid coordinates ($\mathbb{R}$) - $x$ direction")
    return plt.imshow(arr, vmin = -vlim, vmax = vlim, cmap = 'bwr', interpolation = 'none')

def plotPowerSpectrum(model_lr, forcing_variable = "q", save_path = ""):

        # Power spectrum of subgrid potential vorticity
        fig = plt.figure(figsize=(15, 4))
        plt.title(f"Power spectrum of $S_{forcing_variable}$")

        # Retreiving results
        Sq = model_lr.subgrid_forcing('q')

        # Applying fast fourier transform
        line = plt.loglog(*calc_ispec(model_lr.m2, np.abs(model_lr.m2.fft(Sq))[0]**2), label = "Low Resolution")
        plt.loglog(*calc_ispec(model_lr.m2, np.abs(model_lr.m2.fft(Sq))[1]**2), color=line[0]._color, ls='--', label='Low Resolution - (Lower bound)')
        plt.legend(ncol=3)
        plt.grid()
        plt.ylabel("Power spectrum")
        plt.xlabel("Isotropic wavenumber - $\lambda$")

        # Saving results
        save_path_complete = f"{save_path}/subgrid_forcing/power_spectrum/{forcing_variable}"

        # Check if image folder exists
        if not os.path.exists(save_path_complete):
            os.makedirs(save_path_complete)

        # Save the figure
        fig.savefig(save_path_complete + f"/PowerSpectrum_{forcing_variable}.png", bbox_inches = "tight")

def plotForcingVariable(model_lr, forcing_variable = "sq", save_path = ""):

    # Subgrid - Potential vorticity
    if forcing_variable == "sq":

        # Initialization of the figure
        fig = plt.figure(figsize=(22, 6))
        plt.subplot(1, 2, 1, title = '$S_{q_{total}}$')
        plt.ylabel("Grid coordinates ($\mathbb{R}$) - $y$ direction")

        # Total potential vorticity forcing term (Sq_tot)
        imshow_2(model_lr.q_forcing_total[0][0], 3e-11)

        # Subgrid potential vorticity term (Sq)
        plt.subplot(1, 2, 2, title = '$S_{q}$')
        im = imshow_2(model_lr.subgrid_forcing('q')[0], 3e-11)
        cb = fig.colorbar(im, ax = fig.axes, pad=0.15).set_label('$S_{q}$ [$s^{-2}$]')


    # Subgrid - Horizontal velocity
    if forcing_variable == "su":

        # Initialization of the figure
        fig = plt.figure(figsize=(22, 6))
        plt.plot(title = '$S_{u}$')
        plt.ylabel("Grid coordinates ($\mathbb{R}$) - $y$ direction")

        # Subgrid potential vorticity term (Sq)
        im = imshow_2(model_lr.subgrid_forcing('u')[0], 1.5e-7)
        cb = fig.colorbar(im, ax = fig.axes, pad=0.15).set_label('$S_{u}$ [$m\,s^{-2}$]')


    # Subgrid - Vorticity flux
    if forcing_variable == "flux":

        # Initialization of the figure
        fig = plt.figure(figsize=(22, 6))
        plt.subplot(1, 2, 1, title = '$\phi_{q_{u}}$')
        plt.ylabel("Grid coordinates ($\mathbb{R}$) - $y$ direction")

        # Retreiving fluxes
        uq, vq = model_lr.subgrid_fluxes('q')

        # Subgrid vorticity flux in horizontal direction
        imshow_2(uq[1], 1.5e-8)

        # Subgrid vorticity flux in vertical direction
        plt.subplot(1, 2, 2, title = '$\phi_{q_{v}}$')
        im = imshow_2(vq[1], 1.5e-7)
        cb = fig.colorbar(im, ax = fig.axes, pad=0.15).set_label('$\phi_{q}$ [$m\,s^{-2}$]')

    # Complete path to save the figure
    save_path_complete = f"{save_path}/subgrid_forcing/forcing_variable/{forcing_variable}"

    # Check if image folder exists
    if not os.path.exists(save_path_complete):
        os.makedirs(save_path_complete)

    # Save the figure
    fig.savefig(save_path_complete + f"/fv_{forcing_variable}.png", bbox_inches = "tight")