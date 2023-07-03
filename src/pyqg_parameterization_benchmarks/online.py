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

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils             import *
from pyqg_parameterization_benchmarks.utils_TFE         import *
from pyqg_parameterization_benchmarks.plots_TFE         import *
from pyqg_parameterization_benchmarks.online_metrics    import *
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
    USAGE:      python online.py  --folder_online    <X>
                                  --folder_models    <X>
                                  --memory           <X>
                                  --type_sim         <X>
    """
    # Initialization of the parser
    parser = argparse.ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--folder_online',
        help  = 'Folder used to load datasets used for the online test',
        type  = str)

    parser.add_argument(
        '--folder_models',
        help = 'Folder used to load all the models to be tested',
        type = str)

    parser.add_argument(
        '--memory',
        help = 'Total number of memory allocated [GB] (used for security purpose)',
        type = int)

    parser.add_argument(
        '--type_sim',
        help = 'Type of online testing made, i.e. eddies or jets',
        type = str,
        choices = ["EDDIES", "JETS"])

    # Retrieving the values given by the user
    args = parser.parse_args()

    # Display information over terminal (0)
    tfe_title()

    # ----------------------------------
    #              Asserts
    # ----------------------------------
    # Check if the path of each dataset exist
    assert check_datasets_availability([args.folder_online]), \
        f"Assert: One (or more) of the online test dataset does not exist, check again the name of the folders"

    # Check if there is enough memory allocated to load the datasets
    needed_memory = get_datasets_memory([args.folder_online]  , datasets_type = ["LR", "HR"])

    assert math.ceil(needed_memory) < args.memory , \
        f"Assert: Not enough memory allocated to store the offline test datasets ({math.ceil(needed_memory)} [Gb])"

    # Check if the model folder does exist
    assert check_model_folder_availability(args.folder_models), \
        f"Assert: The folder containing all the models to be tested does not exist or is empty"

    # Check if the baseline parameterizations exist
    assert check_model_folder_availability("/___BASELINE___/EDDIES_TRAINING_UNIQUE_5000_50---FCNN_q_to_q_fluxes/"), \
        f"Assert: The folder containing FCNN_q_to_q_fluxes does not exist or is empty"

    assert check_model_folder_availability("/___BASELINE___/EDDIES_TRAINING_UNIQUE_5000_50---FCNN_q_to_q_forcing_total/"), \
        f"Assert: The folder containing FCNN_q_to_q_forcing_total does not exist or is empty"

    assert check_model_folder_availability("/___BASELINE___/EDDIES_TRAINING_UNIQUE_5000_50---FCNN_q_to_q_subgrid_forcing/"), \
        f"Assert: The folder containing FCNN_q_to_q_subgrid_forcing does not exist or is empty"

    # Display information over terminal (1)
    section("Loading datasets")

    # ----------------------------------
    #          Loading datasets
    # ----------------------------------
    data_HR, data_LR, _  = load_data([args.folder_online], datasets_type = ["HR", "LR"])

    # Display information over terminal (2)
    print("\nDone\n")
    section("Loading Trained Parameterizations")

    # ------------------------------------------------------
    #      Accessing & Loading Trained Parameterizations
    # ------------------------------------------------------
    # Name of known parameterizations
    whitelist = ["FCNN", "KASKADE", "UNET", "FNO", "FFNO"]

    # Path to baseline and trained models
    path_models   = "../../models/" + args.folder_models + "/"

    # Listing all the models
    name_models_temp   = os.listdir(path_models)

    # Post-processed list of models name
    name_models = list()

    # Analyzing the models
    for n in name_models_temp:
        for w in whitelist:
            if w in n:
                name_models.append(n)

    # Loading Trained Parameterizations
    parameterizations_models = [NN_Parameterization_Handler(path_models + n) for n in name_models]

    # Display information over terminal (3)
    print("\nDone\n")
    section("Computing / Loading Baseline Parameterizations")

    # ------------------------------------------------------
    #     Accessing & Loading Baseline Parameterizations
    # ------------------------------------------------------
    name_baseline = ["FCNN_q_to_q_forcing_total", "FCNN_q_to_q_subgrid_forcing", "FCNN_q_to_q_fluxes"]

    # Stores simulations results of baseline parameterization
    datasets_baseline = list()

    for i, n in enumerate(name_baseline):

        # Main folder path
        m_path = "../../datasets/" + args.folder_online + "/" + n

        # --- Simulation does not exist ---
        if not os.path.isfile(m_path + "/dataset_LR.nc"):

            # Creation of the folder
            if not os.path.exists(m_path):
                os.makedirs(m_path)

            # Displaying information over terminal (4)
            print("- Generating Simulation Results : " + n + "\n")

            # Loading parameterization
            param = NN_Parameterization_Handler(f"../../models/___BASELINE___/EDDIES_TRAINING_UNIQUE_5000_50---{n}")

            # Generating simulation
            sim = param.run_online(sim_duration       = 10,
                                   skipped_time       = 6 if "JETS" == args.type_sim else 3,
                                   target_sample_size = 1000,
                                   **get_sim_configuration(data_LR))

            # Saving the result in online dataset folder
            sim.to_netcdf(m_path + "/dataset_LR.nc")

            # Adding results for later use
            datasets_baseline.append(sim)

        # --- Simulation exist ---
        else:

            # Displaying information over terminal (5)
            print("- Loading Simulation Results : " + n + "\n")

            # Loading already save dataset
            _, sim, _ = load_data([args.folder_online + "/" + n], datasets_type = ["LR"])

            # Loading already save dataset and saving it for later use
            datasets_baseline.append(sim)

    # Display information over terminal (6)
    print("\nDone\n")
    section("Computing / Loading Analytical Parameterizations")

    # ------------------------------------------------------
    #    Accessing & Loading Analytical Parameterizations
    # ------------------------------------------------------
    name_analyticals = ["Smagorinsky", "BackscatterBiharmonic", "HybridSymbolic"]

    # Initialization of corresponding parameterization
    parameterizations_analytical = [Smagorinsky(0.15), BackscatterBiharmonic(np.sqrt(0.007), 1.2), HybridSymbolic()]

    # Stores simulations results of analytical parameterization
    datasets_analytical = list()

    for i, n in enumerate(name_analyticals):

        # Main folder path
        a_path = "../../datasets/" + args.folder_online + "/" + n

        # --- Simulation does not exist ---
        if not os.path.isfile(a_path + "/dataset_LR.nc"):

            # Creation of the folder
            if not os.path.exists(a_path):
                os.makedirs(a_path)

            # Displaying information over terminal (7)
            print("- Generating Simulation Results : " + n + "\n")

            # Generating simulation
            sim = parameterizations_analytical[i].run_online(sim_duration       = 10,
                                                             skipped_time       = 6 if "JETS" == args.type_sim else 3,
                                                             target_sample_size = 1000,
                                                             **get_sim_configuration(data_LR))

            # Saving the result in online dataset folder
            sim.to_netcdf(a_path + "/dataset_LR.nc")

            # Adding results for later use
            datasets_analytical.append(sim)

        # --- Simulation exist ---
        else:

            # Displaying information over terminal (8)
            print("- Loading Simulation Results : " + n + "\n")

            # Loading already save dataset
            _, sim, _ = load_data([args.folder_online + "/" + n], datasets_type = ["LR"])

            # Loading already save dataset and saving it for later use
            datasets_analytical.append(sim)

    # Display information over terminal (9)
    print("\nDone\n")
    section("Plotting & Comparing")

    # ------------------------------------------------------
    #                  Plotting & Comparing
    # ------------------------------------------------------
    # Used to store simulations results of models to diagnose similarities afterwards
    diagnostic_models = list()

    for name, param in zip(name_models, parameterizations_models):

        # ------------------------------------------------------
        #    Loading / Generating Parameterization Dataset
        # ------------------------------------------------------
        # Path to the current parameterization
        path = "../../datasets/" + args.folder_online + "/" + name + "/"

        # Stores online results of parameterization
        online_results = None

        # --- Simulation does not exist ---
        if not os.path.isfile(path + "/dataset_LR.nc"):

            # Creation of the folder
            if not os.path.exists(path):
                os.makedirs(path)

            # Displaying information over terminal (9)
            print("- Generating Simulation Results : " + name + "\n")

            # Generation of online results using trained parameterization
            try:
                online_results = param.run_online(sim_duration       = 10,
                                                  skipped_time       = 6 if "JETS" == args.type_sim else 3,
                                                  target_sample_size = 1000,
                                                  **get_sim_configuration(data_LR))
            except AssertionError as e:
                print("Online Simulation - CFL criterion violated")
                continue
            except:
                raise


            # Save results for later use
            online_results.to_netcdf(path + "/dataset_LR.nc")

        # --- Simulation does exist ---
        else:

            # Displaying information over terminal (10)
            print("- Loading Simulation Results : " + name + "\n")

            _, online_results, _ = load_data([args.folder_online + "/" + name + "/"], datasets_type = ["LR"])

        # ------------------------------------------------------
        #              Parameterization Diagnostics
        # ------------------------------------------------------
        # Displaying information over terminal (11)
        print("- Computing diagnostics : " + name + "\n")

        # Note : The shape is (diagnostic, name of parameterization)
        diagnostic_models.append((diagnostic_similarities(online_results, target = data_HR, baseline = data_LR),
                                  beautify_model_names([name])[0]))

        # ------------------------------------------------------
        #            Organizing Data For Comparaison
        # ------------------------------------------------------
        simulation_hr         = [(data_HR,                'High-Resolution')]
        simulation_lr         = [(data_LR,                'Low-Resolution')]
        simulation_online     = [(online_results,         beautify_model_names([name])[0])]
        simulation_baseline   = [(datasets_baseline[i],   beautify_model_names([name_baseline[i]])[0]) for i in range(len(datasets_baseline))]
        simulation_analytical = [(datasets_analytical[i], name_analyticals[i]) for i in range(len(datasets_analytical))]

        # Full path to save folder
        save_folder = path_models + f"___ONLINE_{args.type_sim}___"

        # Check if plot folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.makedirs(save_folder + "/EnergyBudget/")
            os.makedirs(save_folder + "/VorticityDistribution/")
            os.makedirs(save_folder + "/DistributionSimilarities/")

        # ------------------------------------------------------
        #                     Energy Budget
        # ------------------------------------------------------
        # Displaying information over terminal (12)
        print("- Energy budget : " + name + "\n")

        fig_1 = energy_budget_figure(simulation_hr + simulation_online + simulation_lr,
                                     simulation_hr + simulation_online + simulation_analytical,
                                     simulation_hr + simulation_online + simulation_baseline)
        plt.tight_layout()
        plt.savefig(save_folder + f"/EnergyBudget/EBF__{name}.svg", bbox_inches = "tight")
        plt.close()

        # ------------------------------------------------------
        #                 Vorticity Distribution
        # ------------------------------------------------------
        # Displaying information over terminal (13)
        print("- Vorticity distribution : " + name + "\n")

        for z in range(2):
            fig_2 = vorticity_distribution_figure(simulation_hr + simulation_lr + simulation_online,
                                                  simulation_analytical,
                                                  simulation_baseline,
                                                  variable = "q",
                                                  level = z)

            plt.tight_layout()
            plt.savefig(save_folder + f"/VorticityDistribution/VDF__z_{str(z)}_{name}.svg", bbox_inches = "tight")
            plt.close()

        # Display information over terminal (14)
        print("\nDone\n")

    # ------------------------------------------------------
    #           Distribution / Spectral Diagnostics
    # ------------------------------------------------------
    # Displaying information over terminal (15)
    print(" - Diagnostic baselines \n")

    diagnostic_baselines   = [(diagnostic_similarities(datasets_baseline[i], target = data_HR, baseline = data_LR),
                              beautify_model_names([name_baseline[i]])[0])
                              for i in range(len(datasets_baseline))]


    # Displaying information over terminal (16)
    print("\nDone\n")
    print(" - Diagnostic analytical \n")

    diagnostic_analyticals = [(diagnostic_similarities(datasets_analytical[i], target = data_HR, baseline = data_LR),
                              name_analyticals[i])
                              for i in range(len(datasets_analytical))]

    # Displaying information over terminal (17)
    print("\nDone\n")
    print(" - Diagnostic figures \n")

    # Plotting diagnostics for each model individually
    for m in diagnostic_models:

        fig_3 = diagnostic_similarities_figure(m, diagnostic_baselines, diagnostic_analyticals)
        plt.tight_layout()
        plt.savefig(save_folder + f"/DistributionSimilarities/DSF_{m[1]}.svg", bbox_inches = "tight")
        plt.close()

    # Display information over terminal (18)
    print("\nDone\n")
