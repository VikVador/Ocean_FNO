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
#    Parameters
# -----------------
"""
results         = ["___OFFLINE_EDDIES__",
                   "___OFFLINE_JETS__",
                   "___OFFLINE_FULL__",
                   "___ONLINE_EDDIES___",
                   "___ONLINE_JETS___"]
"""

results         = ["___ONLINE_EDDIES___",
                   "___ONLINE_JETS___"]


"""
model_folders   = ["__P1__FCNN_UNIQUE_JETS",
                   "__P1__UNET_UNIQUE_EDDIES",
                   "__P1__UNET_UNIQUE_JETS",
                   "__P2__FCNN_MIXED_EDDIES",
                   "__P2__FCNN_MIXED_JETS",
                   "__P2__UNET_MIXED_EDDIES",
                   "__P2__UNET_MIXED_JETS",
                   "__P3__FCNN_MIXED_EDDIES",
                   "__P3__FCNN_MIXED_JETS",
                   "__P4__FCNN_FULL"]
"""
model_folders= ["__P7__ARCH_32_4",
                "__P7__ARCH_32_8",
                "__P7__ARCH_32_12",
                "__P7__ARCH_32_16",
                "__P7__ARCH_32_20",
                "__P7__ARCH_32_24",
                "__P7__ARCH_64_4",
                "__P7__ARCH_64_8",
                "__P7__ARCH_64_12",
                "__P7__ARCH_64_16",
                "__P7__ARCH_64_20",
                "__P7__ARCH_64_24",
                "__P7__ARCH_128_4",
                "__P7__ARCH_128_8",
                "__P7__ARCH_128_12",
                "__P7__ARCH_128_16",
                "__P7__ARCH_128_20",
                "__P7__ARCH_128_24",
                "__P7__ARCH_GOAT__"]

"""
model_folders     = ["__P1__FNO_UNIQUE_EDDIES",
                     "__P1__FNO_UNIQUE_JETS",
                     "__P1__FFNO_UNIQUE_EDDIES",
                     "__P1__FFNO_UNIQUE_JETS",
                     "__P2__FNO_MIXED_EDDIES",
                     "__P2__FNO_MIXED_JETS",
                     "__P2__FFNO_MIXED_EDDIES",
                     "__P2__FFNO_MIXED_JETS",
                     "__P3__FNO_MIXED_EDDIES",
                     "__P3__FNO_MIXED_JETS",
                     "__P3__FFNO_MIXED_EDDIES",
                     "__P3__FFNO_MIXED_JETS",
                     "__P4__FNO_FULL",
                     "__P4__FFNO_FULL"]
"""

# ----------------------------
#         Initialization
# ----------------------------
init = "mkdir update \n"

for m in model_folders:
    init += f"mkdir update/{m}\n"

for m in model_folders:
    for r in results:
        init += f"mkdir update/{m}/{r} \n"

# ----------------------------
#     Downloading results
# ----------------------------
download = ""

for m in model_folders:
    for r in results:
        download += f"scp -r alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/models/{m}/{r} update/{m}/ \n"

# Creating sbatch file for cluster
update_file = open(f"local_update.sh", "w")
update_file.write(init + download)
update_file.close()
