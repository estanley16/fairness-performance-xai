#!/bin/bash
#SBATCH --account=def-nforkert
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=emma.stanley@ucalgary.ca
#SBATCH --output=/home/estanley/scratch/abcd/code/nihpd_to_cerebra.out
module load StdEnv/2020  gcc/9.3.0
module load ants/2.3.5


#register MNI ICBM152 skull stripped atlas to NIHPD symmetric stripped T1
antsRegistrationSyNQuick.sh -d 3 \ #dimensions
                            -f /home/estanley/scratch/abcd/atlas/nihpd_asym_07.5-13.5_t1w_stripped.nii.gz \ #fixed -> NIHPD
                            -m /home/estanley/scratch/abcd/atlas/mni_icbm152_t1_sym_stripped.nii.gz \ #moving -> MNI
                            -t s \ #rigid + affine + deformable transform
                            -o /home/estanley/scratch/abcd/atlas/mni_to_nihpd_ #output prefix
