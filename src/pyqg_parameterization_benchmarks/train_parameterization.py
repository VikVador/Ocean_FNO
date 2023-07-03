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
import math
import argparse
from   torch.utils.tensorboard import SummaryWriter

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils               import *
from pyqg_parameterization_benchmarks.utils_TFE           import *
from pyqg_parameterization_benchmarks.plots_TFE           import *
from pyqg_parameterization_benchmarks.configurations      import *
from pyqg_parameterization_benchmarks.neural_networks     import NN_Parameterization_Handler

# -----------------------------------------------------
#                         Main
# -----------------------------------------------------
if __name__ == '__main__':

    # ----------------------------------
    # Parsing the command-line arguments
    # ----------------------------------
    # Definition of the help message that will be shown on the terminal
    usage = """
    USAGE:      python train_parameterization.py --folder_training    <X>
                                                 --folder_validation  <X>
                                                 --param_name         <X>
                                                 --param_type         <X>
                                                 --inputs             <X>
                                                 --targets            <X>
                                                 --learning_rate      <X>
                                                 --batch_size         <X>
                                                 --optimizer          <X>
                                                 --scheduler          <X>
                                                 --num_epochs         <X>
                                                 --zero_mean          <X>
                                                 --padding            <X>
                                                 --memory             <X>
                                                 --sim_type           <X>
    """
    # Initialization of the parser
    parser = argparse.ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--folder_training',
        help  = 'Folder(s) used to load data as training data',
        nargs = '+')

    parser.add_argument(
        '--folder_validation',
        help  = 'Folder(s) used to load data as validation data',
        nargs = '+')

    parser.add_argument(
        '--param_name',
        help = 'Name of the folder containing the parameterization',
        type = str)

    parser.add_argument(
        '--param_type',
        help = 'Type of parameterization used to learn closure',
        type = str,
        default = "FCNN")

    parser.add_argument(
        '--inputs',
        help  = 'Type of inputs given to the parameterization for training',
        nargs = '+')

    parser.add_argument(
        '--targets',
        help = 'Type of output predicted by the parameterization',
        type = str,
        choices = ['q_forcing_total', 'q_subgrid_forcing', 'q_fluxes'])

    parser.add_argument(
        '--learning_rate',
        help = 'Initial value of the learning rate used during training (it changes with scheduler)',
        type = float,
        default = 0.01)

    parser.add_argument(
        '--batch_size',
        help = 'Size of the batch size used during training',
        type = int,
        default = 64)

    parser.add_argument(
        '--optimizer',
        help = 'Name of the optimizer used during training',
        type = str,
        choices = ["adam", "sgd"],
        default = "adam")

    parser.add_argument(
        '--scheduler',
        help = 'Name of the scheduler used during training',
        type = str,
        choices = ["constant", "multi_step", "exponential", "cosine", "cosine_warmup_restart", "cyclic"],
        default = "constant")

    parser.add_argument(
        '--configuration',
        help = 'The type of model configuration used, i.e. either default or a given number',
        type = str,
        default = "default")

    parser.add_argument(
        '--num_epochs',
        help = 'Number of epochs for training',
        type = int)

    parser.add_argument(
        '--zero_mean',
        help = 'Type of pre-processing made on the datasets (training and validation)',
        type = str,
        default = True)

    parser.add_argument(
        '--padding',
        help = 'Type of padding used by the parameterization',
        type = str,
        choices = ['circular', 'same', 'None'],
        default = 'circular')

    parser.add_argument(
        '--memory',
        help = 'Total number of memory allocated [GB] (used for security purpose)',
        type = int)

    parser.add_argument(
        '--sim_type',
        help = 'Type of fluid simulation studied (used to order tensorboard folders more easily)',
        type = str)

    # Retrieving the values given by the user
    args = parser.parse_args()

    # Display information over terminal (0)
    tfe_title()

    # ----------------------------------
    #              Asserts
    # ----------------------------------
    # Check if the path of each dataset exist
    assert check_datasets_availability(args.folder_training), \
        f"Assert: One (or more) of the traininig dataset does not exist,  check again the name of the folders"

    assert check_datasets_availability(args.folder_validation), \
        f"Assert: One (or more) of the validation dataset does not exist, check again the name of the folders"

    # Check if there is enough memory allocated to load the datasets
    needed_memory = get_datasets_memory(args.folder_training  , datasets_type = ["ALR"]) + \
                    get_datasets_memory(args.folder_validation, datasets_type = ["ALR"])

    assert math.ceil(needed_memory) < args.memory , \
        f"Assert: Not enough memory allocated to store the train and validation datasets ({math.ceil(needed_memory)} [Gb])"

    # Check if the inputs are a combination of q, u, v
    assert check_choices(args.inputs, ["q", "u", "v"]), \
        f"Assert: Input variable(s) for the parameterization must be 'q', 'u', 'v' or a combination"

    # Check that num epoch is positive
    assert 0 < args.num_epochs, \
        f"Assert: Number of epochs must be greater than 0"

    # Check that batch size is positive
    assert 0 < args.batch_size, \
        f"Assert: Batch size must be greater than 0"

    # Update of zero mean argument to bool type
    args.zero_mean = True if args.zero_mean == "True" else False

    # Display information over terminal (1)
    section("Loading datasets")
    show_param_parameters(args)

    if args.configuration != "default":
        show_config_properties(
            get_neural_network_configuration(args.param_name, args.configuration), args.configuration)

    # ----------------------------------
    #          Loading datasets
    # ----------------------------------
    # Training
    _, _, data_ALR_train = load_data(args.folder_training,   datasets_type = ["ALR"])

    # Validation
    _, _, data_ALR_valid = load_data(args.folder_validation, datasets_type = ["ALR"])

    # ----------------------------------
    #     Training parameterization
    # ----------------------------------
    # Write down inputs as string for model name
    input_str = ""
    for index, i in enumerate(args.inputs):
        input_str += i if index == 0 else "_" + i

    # Creation of a dictionnary containing all the training properties used
    training_prop = {
        "lr"            : args.learning_rate,
        "bs"            : args.batch_size,
        "optimizer"     : args.optimizer,
        "scheduler"     : args.scheduler,
        "configuration" : args.configuration,
    }
    # ------ Creation of a complete model name ------
    #
    # Contains the version of the model (from configurations.py)
    version = "---" if args.configuration == "default" else f"---C{args.configuration}---"

    # Model, version & (input, output)
    model_name_part_1 = f"{args.param_name}{version}{input_str}_to_{args.targets}---"

    # Training conditions of the model
    model_name_part_2 = f"E_{args.num_epochs}_LR_{str(args.learning_rate)}_BS_{str(args.batch_size)}_OP_{args.optimizer}_SC_{args.scheduler}"

    # Creation of the complete model name
    curr_model_name = f"{model_name_part_1}{model_name_part_2}---{args.folder_training[0]}"

    # Determine the complete path of the result folder
    model_name, model_path = get_model_path(curr_model_name)

    # Initialization of tensorboard
    tsb = SummaryWriter(f"../../runs/{args.folder_training[0]}/{args.sim_type}/{args.num_epochs}/{args.param_name}{input_str}_to_{args.targets}/{model_name_part_1}{model_name_part_2}")

    # Adaptating targets (if fluxes, needs to predict in u and v directions ! Thus 2 outputs)
    args.targets = [args.targets] if args.targets in ["q_forcing_total", "q_subgrid_forcing"] else ["uq_subgrid_flux", "vq_subgrid_flux"]

    # Training parameterization
    NN_Parameterization_Handler.train_on(dataset            = data_ALR_train,
                                         dataset_validation = data_ALR_valid,
                                         directory          = model_path,
                                         param_name         = args.param_name,
                                         inputs             = args.inputs,
                                         targets            = args.targets,
                                         num_epochs         = args.num_epochs,
                                         zero_mean          = args.zero_mean,
                                         padding            = args.padding,
                                         tensorboard        = tsb,
                                         kw                 = training_prop)
    # Closing tensorboard
    tsb.close()

    # Display information over terminal (2)
    print("\nDone\n")