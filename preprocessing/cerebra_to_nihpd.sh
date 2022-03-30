#!/bin/bash
#SBATCH --account=def-nforkert
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=emma.stanley@ucalgary.ca
#SBATCH --output=/home/estanley/scratch/abcd/atlas/cerebra_to_nihpd.out

#apply deformable registration of cerebra parcellation to nihpd space (using transformation of MNI -> NIHPD)
module load StdEnv/2020  gcc/9.3.0
module load ants/2.3.5
ATLAS_STRIPPED='/home/estanley/scratch/abcd/atlas/nihpd_asym_07.5-13.5_t1w_stripped.nii.gz'
ATLAS_DIR='/home/estanley/scratch/abcd/atlas'

antsApplyTransforms -d 3 \
				-i ${ATLAS_DIR}/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii \
				-r ${ATLAS_STRIPPED} \
               	-o ${ATLAS_DIR}/cerebra_to_nihpd_transformed.nii.gz \
				-n NearestNeighbor \
				-t ${ATLAS_DIR}/mni_to_nihpd_1Warp.nii.gz \
				-t ${ATLAS_DIR}/mni_to_nihpd_0GenericAffine.mat \
				-v 1 \
