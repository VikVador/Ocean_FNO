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
import math
import argparse 
import numpy                 as np
import pandas                as pd
import seaborn               as sns
import matplotlib.pyplot     as plt
from torch.utils.tensorboard import SummaryWriter

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils                  import *
from pyqg_parameterization_benchmarks.utils_TFE              import *
from pyqg_parameterization_benchmarks.plots_TFE              import *
from pyqg_parameterization_benchmarks.neural_networks        import FullyCNN, FCNNParameterization
from pyqg_parameterization_benchmarks.kaskade                import Kaskade,  KASKADEParameterization

# -----------------------------------------------------
#                         Main
# -----------------------------------------------------
if __name__ == '__main__':

    # ----------------------------------
    # Parsing the command-line arguments
    # ----------------------------------
    # Definition of the help message that will be shown on the terminal
    usage = """
    USAGE:      python offline.py --folder_offline   <X>
                                  --folder_models    <X>
                                  --memory           <X>
    """
    # Initialization of the parser
    parser = argparse.ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--folder_offline',
        help  = 'Folder used to load datasets used for the offline test',
        nargs = '+')
    
    parser.add_argument(
        '--folder_models',
        help = 'Folder used to load all the models to be tested',
        type = str,
        default = "base")

    parser.add_argument(
        '--memory',
        help = 'Total number of memory allocated [GB] (used for security purpose)',
        type = int)
    
    # Retrieving the values given by the user
    args = parser.parse_args()

    # Display information over terminal (0)
    tfe_title()
        
    # ----------------------------------
    #              Asserts
    # ----------------------------------
    # Check if the path of each dataset exist
    assert check_datasets_availability(args.folder_offline), \
        f"Assert: One (or more) of the offline test dataset does not exist,  check again the name of the folders"
    
    # Check if there is enough memory allocated to load the datasets
    needed_memory = get_datasets_memory(args.folder_offline  , datasets_type = ["ALR"])
    
    assert math.ceil(needed_memory) < args.memory , \
        f"Assert: Not enough memory allocated to store the offline test datasets ({math.ceil(needed_memory)} [Gb])"

    # Check if the model folder does exist
    assert check_model_folder_availability(args.folder_models), \
        f"Assert: The folder containing all the models to be tested does not exist or is empty"

    # Display information over terminal (1)
    section("Loading datasets")

    # ----------------------------------
    #          Loading datasets
    # ----------------------------------
    _, _, data_ALR_offline  = load_data(args.folder_offline, datasets_type = ["ALR"])
    
    # Display information over terminal (2)
    section("Loading parameterizations")
    
    # ----------------------------------
    #     Accessing parameterizations
    # ----------------------------------
    # Prefix of known parameterizations
    whitelist = ["FCNN", "KASKADE"]
    
    # Stores the type of parameterization tested (based on prefix)
    prefix_parameterizations = []
    
    # Complete path to model folder
    path_model_folder = "../../models/" + args.folder_models + "/"
    
    # Retreives a list of all models
    models_name_c = os.listdir(path_model_folder)
    
    # Cleaning list of model names (Removing jupyter checkpoints, ...)
    models_name = list()
    for n in models_name_c:
        for w in whitelist:

            # Checks if the name is related to a parameterization
            if w in n:
                models_name.append(n)
                prefix_parameterizations.append(w)
                break
    
    # ----------------------------------
    #      Loading parameterizations
    # ----------------------------------
    # Stores all the parametrizations to test offline
    parameterizations = []

    for name, prefix in zip(models_name, prefix_parameterizations):
        
        # Path to model
        full_path = path_model_folder + name
        
        # Loading
        if prefix == "FCNN":
            parameterizations.append(FCNNParameterization(full_path))
            
        if prefix == "KASKADE":
            parameterizations.append(KASKADEParameterization(full_path))
        
    # Display information over terminal (3)
    section("Offline testing")
    show_models_offline(args.folder_models, args.folder_offline, models_name)
    
    # ----------------------------------
    #               Testing
    # ----------------------------------
    # Retreive parmetrization types (i.e. what they predict. Ex: Sq_tot, Sq_forcing, ...)
    param_targets = get_param_type(models_name)
    
    # Stores the different R^2 and corr values measured
    r2_z0, r2_z1, corr_z0, corr_z1 = list(), list(), list(), list()
    
    # Display information over terminal (4)
    print("\nComputing (done)    =")
    
    for param, param_name, param_target, prefix in zip(parameterizations, models_name, param_targets, prefix_parameterizations):
        
        # Computing the predictions made by the parameterization
        offline_pred = param.test_offline(data_ALR_offline, prefix)

        # Plotting the results
        fig = plt.figure(figsize = (8, 6))

        # Adding title
        plt.suptitle(f"Offline performance")
        
        # --- Correlation ---
        for z in [0, 1]:
            
            plt.subplot(2, 2, z + 1, title = f"{['Upper','Lower'][z]} layer")
            
            # --- Level 0 ---
            if z == 0:
                if param_target == 0:
                    corr_z0.append(imshow_offline(offline_pred.q_subgrid_forcing_spatial_correlation.isel(lev = z)))
                if param_target == 1:
                    corr_z0.append(imshow_offline(offline_pred.q_forcing_total_spatial_correlation.isel(lev = z)))
                if param_target == 2:
                    corr_z0.append(imshow_offline(offline_pred.uq_subgrid_flux_spatial_correlation.isel(lev = z)))
                if param_target == 3:
                    corr_z0.append(imshow_offline(offline_pred.vq_subgrid_flux_spatial_correlation.isel(lev = z)))
            
            # --- Level 1 ---
            if z == 1:
                if param_target == 0:
                    corr_z1.append(imshow_offline(offline_pred.q_subgrid_forcing_spatial_correlation.isel(lev = z)))
                if param_target == 1:
                    corr_z1.append(imshow_offline(offline_pred.q_forcing_total_spatial_correlation.isel(lev = z)))
                if param_target == 2:
                    corr_z1.append(imshow_offline(offline_pred.uq_subgrid_flux_spatial_correlation.isel(lev = z)))
                if param_target == 3:
                    corr_z1.append(imshow_offline(offline_pred.vq_subgrid_flux_spatial_correlation.isel(lev = z)))

            if z: colorbar("ρ")

        # --- R^2 ---
        for z in [0, 1]:
                                   
            plt.subplot(2, 2, z + 3)
                                     
            # --- Level 0 ---
            if z == 0:
                if param_target == 0:
                    r2_z0.append(imshow_offline(offline_pred.q_subgrid_forcing_spatial_correlation.isel(lev = z)))
                if param_target == 1:
                    r2_z0.append(imshow_offline(offline_pred.q_forcing_total_spatial_correlation.isel(lev = z)))
                if param_target == 2:
                    r2_z0.append(imshow_offline(offline_pred.uq_subgrid_flux_spatial_correlation.isel(lev = z)))
                if param_target == 3:
                    r2_z0.append(imshow_offline(offline_pred.vq_subgrid_flux_spatial_correlation.isel(lev = z)))
            
            # --- Level 1 ---
            if z == 1:
                if param_target == 0:
                    r2_z1.append(imshow_offline(offline_pred.q_subgrid_forcing_spatial_skill.isel(lev = z)))
                if param_target == 1:
                    r2_z1.append(imshow_offline(offline_pred.q_forcing_total_spatial_skill.isel(lev = z)))  
                if param_target == 2:
                    r2_z1.append(imshow_offline(offline_pred.uq_subgrid_flux_spatial_correlation.isel(lev = z)))
                if param_target == 3:
                    r2_z1.append(imshow_offline(offline_pred.vq_subgrid_flux_spatial_correlation.isel(lev = z)))    
                             
            if z: colorbar("$R^2$")
                
        # Check if plot folder exists
        if not os.path.exists(path_model_folder + "___PLOTS___/"):
            os.makedirs(path_model_folder + "___PLOTS___/")

        # Save the figure
        fig.savefig(path_model_folder + "___PLOTS___/" + f"{param_name}.svg")
        
        # Close current figure
        plt.close(fig)
        
        # Display information over terminal (5)
        print("\n                      - " + param_name)
    
    # Metrics transformed into pandas dataframe for ease of making comparison plot
    metrics = pd.DataFrame({
        "Model"   : beautify_model_names(models_name),
        "R2_z0"   : r2_z0,
        "R2_z1"   : r2_z1,
        "corr_z0" : corr_z0,
        "corr_z1" : corr_z1,
    })
    
    # Metrics conversion to float
    metrics['R2_z0']   = metrics['R2_z0'].astype(float)
    metrics['R2_z1']   = metrics['R2_z1'].astype(float)
    metrics['corr_z0'] = metrics['corr_z0'].astype(float)
    metrics['corr_z1'] = metrics['corr_z1'].astype(float)
    
    # Display information over terminal (6)
    print("\n")
    section("Result comparaison")
    
    # ----------------------------------
    #            Comparaison plot
    # ----------------------------------  
    # Simple tweak of overall plot looks
    sns.set_theme()
    sns.set_context('notebook')

    # Creation of comparaison plots for each metric
    comparaison_plot_offline(metrics, path_model_folder, x = "Model", y = 'R2_z0',   ylab = "$R^2$ ($Z$ = 0)", \
                             min_v = min(r2_z0), max_v = max(r2_z0))
    
    comparaison_plot_offline(metrics, path_model_folder, x = "Model", y = 'R2_z1',   ylab = "$R^2$ ($Z$ = 1)", \
                             min_v = min(r2_z1), max_v = max(r2_z1))

    comparaison_plot_offline(metrics, path_model_folder, x = "Model", y = 'corr_z0', ylab = "ρ ($Z$ = 0)", \
                             min_v = min(corr_z0), max_v = max(corr_z0))

    comparaison_plot_offline(metrics, path_model_folder, x = "Model", y = 'corr_z1', ylab = "ρ ($Z$ = 1)", \
                             min_v = min(corr_z1), max_v = max(corr_z1))

    # Display information over terminal (7)
    print("\nDone\n")
