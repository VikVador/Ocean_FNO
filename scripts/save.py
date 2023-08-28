# Parameters to explore
modes_low_pass  = [(0, 8),  (0, 16), (0, 24), (0, 32)]
modes_pass_band = [(8, 16), (16, 24), (24, 32)]
modes           = modes_low_pass + modes_pass_band
width           = [32, 64, 128]
layers          = [4, 8, 12, 16, 20, 24]
weights         = [True]

# Total number of combinations possible (here 48 in total)
total_comb = len(modes) * len(width) * len(layers) * len(weights)

# Keep count of the configuration index
config_index = 0

# Command file
command_file = ""

for we in weights:
    for w in width:
        for i, l in enumerate(layers):

            # Creation of the folder
            folder_name = f"__P7__ARCH_{w}_{l}"

            # Adding command
            command_file += f"\"{folder_name}\",\n"

            for m in modes:


                config_index = config_index + 1


# Opening file
job_file = open(f"goat.sh", "w")
job_file.write(command_file)
job_file.close()
