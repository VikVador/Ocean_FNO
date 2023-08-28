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
import matplotlib.gridspec   as gridspec

from matplotlib.colorbar     import Colorbar
from torch.utils.tensorboard import SummaryWriter

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils             import *
from pyqg_parameterization_benchmarks.utils_TFE         import *
from pyqg_parameterization_benchmarks.plots_TFE         import *
from pyqg_parameterization_benchmarks.neural_networks   import NN_Parameterization_Handler
from pyqg_parameterization_benchmarks.nn_analytical     import BackscatterBiharmonic, Smagorinsky, HybridSymbolic



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
    
    parser.add_argument(
        '--type_sim',
        help = 'Type of offline testing made, i.e. eddies or jets',
        type = str)
    
    # Retrieving the values given by the user
    args = parser.parse_args()

    # Display information over terminal (0)
    tfe_title()
        
    # ----------------------------------
    #              Asserts
    # ----------------------------------
    # Check if the path of each dataset exist
    assert check_datasets_availability(args.folder_offline), \
        f"Assert: One (or more) of the offline test dataset does not exist, check again the name of the folders"
    
    # Check if there is enough memory allocated to load the datasets
    needed_memory = get_datasets_memory(args.folder_offline  , datasets_type = ["ALR"])
    
    assert math.ceil(needed_memory) < args.memory , \
        f"Assert: Not enough memory allocated to store the offline test datasets ({math.ceil(needed_memory)} [Gb])"

    # Check if the model folder does exist
    assert check_model_folder_availability(args.folder_models), \
        f"Assert: The folder containing all the models to be tested does not exist or is empty"
    
    # Check if the baseline folder does exist
    assert check_model_folder_availability("/___BASELINE___/"), \
        f"Assert: The folder containing all the baseline models to be tested does not exist or is empty"

    # Display information over terminal (1)
    section("Loading datasets")

    # ----------------------------------
    #          Loading datasets
    # ----------------------------------
    _, _, data_ALR_offline  = load_data(args.folder_offline, datasets_type = ["ALR"])
    
    # Display information over terminal (2)
    print("\nDone\n")
    section("Loading parameterizations")
    
    # ------------------------------------------------------------------
    #      Accessing & Loading Trained / Baseline Parameterizations
    # ------------------------------------------------------------------
    # Name of known parameterizations
    whitelist = ["FCNN", "KASKADE", "UNET", "FFNO", "FNO"]

    # Path to baseline and trained models
    path_models   = "../../models/" + args.folder_models + "/"
    path_baseline = "../../models/___BASELINE___/"

    # Listing all the models
    name_models_temp   = os.listdir(path_models)
    name_baseline_temp = os.listdir(path_baseline)

    # Post-processed list of models name
    name_models, name_baseline_nn = list(), list()

    # Analyzing the models
    for n in name_models_temp:
        for w in whitelist:
            if w in n:
                name_models.append(n)
                break

    # Analyzing the baseline           
    for n in name_baseline_temp:
        for w in whitelist:
            if w in n:
                name_baseline_nn.append(n)

    # Loading Trained / Baseline Parameterizations
    parameterizations_models       = [NN_Parameterization_Handler(path_models   + n) for n in name_models]
    parameterizations_baseline_NN  = [NN_Parameterization_Handler(path_baseline + n) for n in name_baseline_nn]
    
    # --------------------------------------------------------
    #             Computing Baseline Predictions
    # --------------------------------------------------------
    offline_preds_baseline_nn = [p.test_offline(data_ALR_offline) for p in parameterizations_baseline_NN]
    
    # --------------------------------------------------------
    #          Plotting & Comparing (1 Model VS Baseline)
    # --------------------------------------------------------
    # Stores the different R^2 and correlation values measured
    r2_z0, r2_z1, corr_z0, corr_z1 = list(), list(), list(), list()

    # Display information over terminal (4)
    print("\nComputing (done)    =")

    # Analyzing the different models 
    for param, name in zip(parameterizations_models, name_models):

        # Determine the number of lines needed to make subplot
        lines = sum([1 for n in [name] + name_baseline_nn])
        
        # Keeps the count of the plot number
        plt_id = 0

        # Creation of a new model figure
        fig = plt.figure(figsize = (12, 12))

        # Used to plot results in grid shape
        gs = gridspec.GridSpec(lines, 6, width_ratios = [1,1,1,1,0.02,0.2])

        # Set the spacing between axes 
        gs.update(wspace = 0.025, hspace = 0.15)

        # Used to determine which prediction load for baseline
        b_index = 0

        # Comparing model & baseline
        for p, n in zip([param] + parameterizations_baseline_NN, [name] + name_baseline_nn):

            # Computing prediction
            offline_pred =  p.test_offline(data_ALR_offline) if n not in name_baseline_nn else \
                            offline_preds_baseline_nn[b_index]

            # Updating the baseline index
            b_index = b_index if n not in name_baseline_nn else b_index + 1

            # Updating color map
            cmap_color = "inferno" if b_index == 0 else "inferno"

            # Determine the type of target predicted by the parameterization
            target_type = list()

            if "q_subgrid_forcing" in n:
                target_type = ["q_subgrid_forcing"]
            elif "q_forcing_total" in n:
                target_type = ["q_forcing_total"]
            elif "q_fluxes"        in n:
                target_type = ["uq_subgrid_flux", "vq_subgrid_flux"]
            else:
                raise Exception("ERROR (Offline.py), unknown target type")

            # Stores final predictions
            corr_z0_pred, corr_z1_pred, r2_z0_pred, r2_z1_pred = list(), list(), list(), list()
                
            # Plotting
            for i, t in enumerate(target_type):
                
                # Computing results
                corr_z0_pred.append(offline_pred[f"{t}_spatial_correlation"].isel(lev = 0))
                corr_z1_pred.append(offline_pred[f"{t}_spatial_correlation"].isel(lev = 1))
                r2_z0_pred.append(  offline_pred[f"{t}_spatial_skill"].isel(      lev = 0))
                r2_z1_pred.append(  offline_pred[f"{t}_spatial_skill"].isel(      lev = 1))
                
                # If fluxes, skipping to compute mean !
                if len(target_type) == 2 and i == 0:
                    continue
                
                # Beautifying the plot (1)
                plt.subplot(gs[plt_id, 0])
                plt.axis('on')
                plt.title(r"Upper $\rho$" if plt_id == 0 else "")
                plt.ylabel(f"{beautify_model_names([n])[0]}", rotation = 0, fontsize = 12, labelpad = 180)

                # Correlation z = 0
                corr_z0.append(imshow_offline(corr_z0_pred, cmap_color = cmap_color)) if b_index == 0 else \
                               imshow_offline(corr_z0_pred, cmap_color = cmap_color)

                # Beautifying the plot (2)
                plt.subplot(gs[plt_id, 1])
                plt.title(r"Lower $\rho$" if plt_id == 0 else "")

                # Correlation z = 1
                corr_z1.append(imshow_offline(corr_z1_pred, cmap_color = cmap_color)) if b_index == 0 else \
                               imshow_offline(corr_z1_pred, cmap_color = cmap_color)

                # Beautifying the plot (3)
                plt.subplot(gs[plt_id, 2])
                plt.title(r"Upper $R^2$" if plt_id == 0 else "")

                # RMSE z = 0
                r2_z0.append(imshow_offline(r2_z0_pred, cmap_color = cmap_color)) if b_index == 0 else \
                             imshow_offline(r2_z0_pred, cmap_color = cmap_color)

                # Beautifying the plot (4)
                plt.subplot(gs[plt_id, 3])
                plt.title(r"Lower $R^2$" if plt_id == 0 else "")

                # RMSE z = 1
                r2_z1.append(imshow_offline(r2_z1_pred, cmap_color = cmap_color)) if b_index == 0 else \
                             imshow_offline(r2_z1_pred, cmap_color = cmap_color)

                # Adding colorbar to model results
                if b_index == 0:

                    # Plot for colorbar
                    cbarax = plt.subplot(gs[0, -1])
                    plt.colorbar(cax = cbarax)

                # Updating id for newt row of plots
                plt_id = plt_id + 1

        # Final processing
        plt.subplots_adjust(wspace = 0.025, hspace = 0.005)

        # Final colorbar
        cbarax = plt.subplot(gs[1:, -1])
        plt.colorbar(cax = cbarax)

        # Plot folder name
        pltf_name = f"___OFFLINE_{args.type_sim}__/"

        # Check if plot folder exists
        if not os.path.exists(path_models + pltf_name):
            os.makedirs(path_models + pltf_name)

        # Save the figure
        plt.savefig(path_models + pltf_name + f"{name}.svg", bbox_inches = "tight")

        # Display information over terminal (5)
        print("\n                      - " + name)

        # Close current figure
        plt.close()
        
    # Metrics transformed into pandas dataframe for ease of making comparison plot
    metrics = pd.DataFrame({
        "Model"   : beautify_model_names(name_models),
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
    comparaison_plot_offline(metrics, path_models + pltf_name, x = "Model", y = 'R2_z0',   ylab = "$R^2$ ($Z$ = 0)", \
                             min_v = min(r2_z0), max_v = max(r2_z0))

    comparaison_plot_offline(metrics, path_models + pltf_name, x = "Model", y = 'R2_z1',   ylab = "$R^2$ ($Z$ = 1)", \
                             min_v = min(r2_z1), max_v = max(r2_z1))

    comparaison_plot_offline(metrics, path_models + pltf_name, x = "Model", y = 'corr_z0', ylab = "ρ ($Z$ = 0)", \
                             min_v = min(corr_z0), max_v = max(corr_z0))

    comparaison_plot_offline(metrics, path_models + pltf_name, x = "Model", y = 'corr_z1', ylab = "ρ ($Z$ = 1)", \
                             min_v = min(corr_z1), max_v = max(corr_z1))

    # Display information over terminal (7)
    print("\nDone\n")
