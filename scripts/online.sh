#!/usr/bin/env bash
#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# Move to correct folder
cd online

sbatch __P1__FNO_UNIQUE_EDDIES_EDDIES_ONLINE.sh
sbatch __P1__FNO_UNIQUE_EDDIES_JETS_ONLINE.sh
sbatch __P1__FNO_UNIQUE_JETS_EDDIES_ONLINE.sh
sbatch __P1__FNO_UNIQUE_JETS_JETS_ONLINE.sh
sbatch __P1__FFNO_UNIQUE_EDDIES_EDDIES_ONLINE.sh
sbatch __P1__FFNO_UNIQUE_EDDIES_JETS_ONLINE.sh
sbatch __P1__FFNO_UNIQUE_JETS_EDDIES_ONLINE.sh
sbatch __P1__FFNO_UNIQUE_JETS_JETS_ONLINE.sh
sbatch __P2__FNO_MIXED_EDDIES_EDDIES_ONLINE.sh
sbatch __P2__FNO_MIXED_EDDIES_JETS_ONLINE.sh
sbatch __P2__FNO_MIXED_JETS_EDDIES_ONLINE.sh
sbatch __P2__FNO_MIXED_JETS_JETS_ONLINE.sh
sbatch __P2__FFNO_MIXED_EDDIES_EDDIES_ONLINE.sh
sbatch __P2__FFNO_MIXED_EDDIES_JETS_ONLINE.sh
sbatch __P2__FFNO_MIXED_JETS_EDDIES_ONLINE.sh
sbatch __P2__FFNO_MIXED_JETS_JETS_ONLINE.sh
sbatch __P3__FNO_MIXED_EDDIES_EDDIES_ONLINE.sh
sbatch __P3__FNO_MIXED_EDDIES_JETS_ONLINE.sh
sbatch __P3__FNO_MIXED_JETS_EDDIES_ONLINE.sh
sbatch __P3__FNO_MIXED_JETS_JETS_ONLINE.sh
sbatch __P3__FFNO_MIXED_EDDIES_EDDIES_ONLINE.sh
sbatch __P3__FFNO_MIXED_EDDIES_JETS_ONLINE.sh
sbatch __P3__FFNO_MIXED_JETS_EDDIES_ONLINE.sh
sbatch __P3__FFNO_MIXED_JETS_JETS_ONLINE.sh
sbatch __P4__FNO_FULL_EDDIES_ONLINE.sh
sbatch __P4__FNO_FULL_JETS_ONLINE.sh
sbatch __P4__FFNO_FULL_EDDIES_ONLINE.sh
sbatch __P4__FFNO_FULL_JETS_ONLINE.sh
