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
cd offline

sbatch __P1__FNO_UNIQUE_EDDIES_EDDIES_OFFLINE.sh
sbatch __P1__FNO_UNIQUE_EDDIES_JETS_OFFLINE.sh
sbatch __P1__FNO_UNIQUE_EDDIES_FULL_OFFLINE.sh
sbatch __P1__FNO_UNIQUE_JETS_EDDIES_OFFLINE.sh
sbatch __P1__FNO_UNIQUE_JETS_JETS_OFFLINE.sh
sbatch __P1__FNO_UNIQUE_JETS_FULL_OFFLINE.sh
sbatch __P1__FFNO_UNIQUE_EDDIES_EDDIES_OFFLINE.sh
sbatch __P1__FFNO_UNIQUE_EDDIES_JETS_OFFLINE.sh
sbatch __P1__FFNO_UNIQUE_EDDIES_FULL_OFFLINE.sh
sbatch __P1__FFNO_UNIQUE_JETS_EDDIES_OFFLINE.sh
sbatch __P1__FFNO_UNIQUE_JETS_JETS_OFFLINE.sh
sbatch __P1__FFNO_UNIQUE_JETS_FULL_OFFLINE.sh
sbatch __P2__FNO_MIXED_EDDIES_EDDIES_OFFLINE.sh
sbatch __P2__FNO_MIXED_EDDIES_JETS_OFFLINE.sh
sbatch __P2__FNO_MIXED_EDDIES_FULL_OFFLINE.sh
sbatch __P2__FNO_MIXED_JETS_EDDIES_OFFLINE.sh
sbatch __P2__FNO_MIXED_JETS_JETS_OFFLINE.sh
sbatch __P2__FNO_MIXED_JETS_FULL_OFFLINE.sh
sbatch __P2__FFNO_MIXED_EDDIES_EDDIES_OFFLINE.sh
sbatch __P2__FFNO_MIXED_EDDIES_JETS_OFFLINE.sh
sbatch __P2__FFNO_MIXED_EDDIES_FULL_OFFLINE.sh
sbatch __P2__FFNO_MIXED_JETS_EDDIES_OFFLINE.sh
sbatch __P2__FFNO_MIXED_JETS_JETS_OFFLINE.sh
sbatch __P2__FFNO_MIXED_JETS_FULL_OFFLINE.sh
sbatch __P3__FNO_MIXED_EDDIES_EDDIES_OFFLINE.sh
sbatch __P3__FNO_MIXED_EDDIES_JETS_OFFLINE.sh
sbatch __P3__FNO_MIXED_EDDIES_FULL_OFFLINE.sh
sbatch __P3__FNO_MIXED_JETS_EDDIES_OFFLINE.sh
sbatch __P3__FNO_MIXED_JETS_JETS_OFFLINE.sh
sbatch __P3__FNO_MIXED_JETS_FULL_OFFLINE.sh
sbatch __P3__FFNO_MIXED_EDDIES_EDDIES_OFFLINE.sh
sbatch __P3__FFNO_MIXED_EDDIES_JETS_OFFLINE.sh
sbatch __P3__FFNO_MIXED_EDDIES_FULL_OFFLINE.sh
sbatch __P3__FFNO_MIXED_JETS_EDDIES_OFFLINE.sh
sbatch __P3__FFNO_MIXED_JETS_JETS_OFFLINE.sh
sbatch __P3__FFNO_MIXED_JETS_FULL_OFFLINE.sh
sbatch __P4__FNO_FULL_EDDIES_OFFLINE.sh
sbatch __P4__FNO_FULL_JETS_OFFLINE.sh
sbatch __P4__FNO_FULL_FULL_OFFLINE.sh
sbatch __P4__FFNO_FULL_EDDIES_OFFLINE.sh
sbatch __P4__FFNO_FULL_JETS_OFFLINE.sh
sbatch __P4__FFNO_FULL_FULL_OFFLINE.sh

