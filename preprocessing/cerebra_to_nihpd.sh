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
ATLAS_STRIPPED='/home/estanley/scratch/abcd/atlas/nihpd_asym_07.5-13.5_t1w_stripped.nii.gz' #NIHPD atlas
ATLAS_DIR='/home/estanley/scratch/abcd/atlas'

antsApplyTransforms -d 3 \ #3 dimensions
				-i ${ATLAS_DIR}/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii \ #input: cerebrA
				-r ${ATLAS_STRIPPED} \ #reference: NIHPD
               	-o ${ATLAS_DIR}/cerebra_to_nihpd_transformed.nii.gz \ #output filename
				-n NearestNeighbor \ #interpolation method
				-t ${ATLAS_DIR}/mni_to_nihpd_1Warp.nii.gz \ #deformation field to apply
				-t ${ATLAS_DIR}/mni_to_nihpd_0GenericAffine.mat \ #affine transformation matrix to apply
				-v 1 \ #verbosity
