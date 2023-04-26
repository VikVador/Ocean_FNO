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

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils_TFE import *

# -----------------------------------------------------
#                         Main
# -----------------------------------------------------
if __name__ == '__main__':
    
    # ----------------------------------
    #           Loading data
    # ----------------------------------
    # Datasets to be loaded
    training_folder = [f"EDDIES_TRAINING_MIXED_{i}" for i in range(1, 6)]  + [f"JETS_TRAINING_MIXED_{i}" for i in range(1, 6)]      # FULL 5000
    #training_folder = ["EDDIES_TRAINING_MIXED_5000",  "JETS_TRAINING_MIXED_5000"]                                                   # FULL 10000
    #training_folder = ["EDDIES_TRAINING_MIXED_10000", "JETS_TRAINING_MIXED_10000"]                                                  # FULL 20000
    #training_folder = ["EDDIES_TRAINING_MIXED_20000", "JETS_TRAINING_MIXED_20000"]                                                  # FULL 40000
    #training_folder = [f"EDDIES_VALIDATION_MIXED_{i}" for i in range(1, 6)]  + [f"JETS_VALIDATION_MIXED_{i}" for i in range(1, 6)]  # Validation 5000
    #training_folder = [f"EDDIES_OFFLINE_MIXED_{i}" for i in range(1, 6)]     + [f"JETS_OFFLINE_MIXED_{i}" for i in range(1, 6)]     # Offline    5000
    
    # Name of the final folder
    saving_folder = "FULL_TRAINING_MIXED_5000"
    #saving_folder = "FULL_TRAINING_MIXED_10000"
    #saving_folder = "FULL_TRAINING_MIXED_20000"
    #saving_folder = "FULL_TRAINING_MIXED_40000"
    #saving_folder = "FULL_VALIDATION"
    #saving_folder = "FULL_OFFLINE"
    
    # Displaying information over terminal
    print("Folder loaded : ")
    for f in training_folder:
        print("- ", f)
        
    print("Saving folder : ", saving_folder)
    
    # Loading the data
    data_HR, data_LR, data_ALR = load_data(training_folder, datasets_type = ["HR", "LR", "ALR"])
    
    print("Data loaded")
    print("-------- HR --------")
    print(data_HR)
    print("-------- LR --------")
    print(data_LR)
    print("-------- ALR --------")
    print(data_ALR)
    
    # ----------------------------------
    #          Saving datasets
    # ----------------------------------
    # Complete path to saving folder
    saving_path = "../../datasets/" + saving_folder

    # Checks if the saving folder exists, if not, creates it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # Saving everything to special format (recommended by xarray doccumentation)
    data_HR.to_netcdf( saving_path + "/dataset_HR.nc")
    data_ALR.to_netcdf(saving_path + "/dataset_ALR.nc" )
    data_LR.to_netcdf( saving_path + "/dataset_LR.nc" )
    
    # Display information over terminal (5)
    print("\nDone\n")