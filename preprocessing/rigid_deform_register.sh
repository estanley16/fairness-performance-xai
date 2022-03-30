#!/bin/bash


#script for skull stripping and rigid registration of patient scans to nihpd atlas
#also applies nonlinear transformation of rigid registered scans to atlas to use the transformation matrix in downstream processing of saliency maps


#set directories
DATAPATH='/home/estanley/scratch/abcd_miccai/data'
ATLAS='/home/estanley/scratch/abcd_miccai/atlas/nihpd_asym_07.5-13.5_t1w.nii.gz'
ATLAS_MASK='/home/estanley/scratch/abcd_miccai/atlas/nihpd_asym_07.5-13.5_mask.nii.gz'
OUTPUT_DIR='/home/estanley/scratch/abcd_miccai/rigid_processed'
ATLAS_STRIPPED='/home/estanley/scratch/abcd_miccai/atlas/nihpd_asym_07.5-13.5_t1w_stripped.nii.gz'

#if output directory doesn't exist, make it
[[ -d "${OUTPUT_DIR}" ]] || mkdir "${OUTPUT_DIR}"

#loop through files in the dataset
for IMG in `ls $DATAPATH/*.nii.gz`
do
	FILENAME=$(cut -f 7 -d "/" <<<"$IMG")
        SUBJECT_ID=$(cut -c 5-19 <<<"$FILENAME")

	echo "*************************************************************"
	echo " 			${SUBJECT_ID}: creating file				       "
	echo "*************************************************************"

	OUTPUT_PATH="${OUTPUT_DIR}/${SUBJECT_ID}"
	mkdir -p "${OUTPUT_DIR}/${SUBJECT_ID}/".

	cat >rigidreg_${SUBJECT_ID}.sh <<-EOF1
	#!/bin/bash
	#SBATCH --account=def-nforkert
	#SBATCH --time=00:45:00
	#SBATCH --cpus-per-task=1
	#SBATCH --mem-per-cpu=6G
	#SBATCH --mail-type=FAIL
	#SBATCH --mail-user=emma.stanley@ucalgary.ca
	#SBATCH --output=/home/estanley/scratch/abcd_miccai/code/rigid_output/${SUBJECT_ID}.out
	module load StdEnv/2020  gcc/9.3.0
	module load ants/2.3.5

	echo "*************************************************************"
	echo " 			${SUBJECT_ID}: Processing has begun!				       "
	echo "*************************************************************"

	#skull strip by deforming atlas mask to each T1 image and then multiplying, then rigid registration using transformed mask

	echo "Registering atlas to ${SUBJECT_ID} T1..."
        ${ANTSPATH}antsRegistrationSyNQuick.sh -d 3 \
						-f ${IMG} \
						-m ${ATLAS} \
						-t s \
						-o ${OUTPUT_PATH}/${SUBJECT_ID}_atlas_registered_ \

	echo "Transforming template mask into patient specific mask using atlas transformations..."
	${ANTSPATH}antsApplyTransforms -d 3 \
					-i ${ATLAS_MASK} \
					-r ${IMG} \
                   			-o ${OUTPUT_PATH}/${SUBJECT_ID}_mask_transformed.nii.gz \
					-n NearestNeighbor \
					-t ${OUTPUT_PATH}/${SUBJECT_ID}_atlas_registered_1Warp.nii.gz \
					-t ${OUTPUT_PATH}/${SUBJECT_ID}_atlas_registered_0GenericAffine.mat \
					-v 1 \



	echo "Using mask to skull strip..."
	#format: dimensions (3), output T1, multiply (m), inputs: transformed mask, input T1 image
	#output is the original T1 image, skull stripped
        ${ANTSPATH}ImageMath 3 ${OUTPUT_PATH}/${SUBJECT_ID}_raw_T1strip.nii.gz m ${OUTPUT_PATH}/${SUBJECT_ID}_mask_transformed.nii.gz ${IMG}


	echo "Applying rigid registration..."
	#rigidly register skull stripped image to skull stripped atlas
        ${ANTSPATH}antsRegistrationSyNQuick.sh -d 3 \
						-f ${ATLAS_STRIPPED} \
						-m ${OUTPUT_PATH}/${SUBJECT_ID}_raw_T1strip.nii.gz \
						-t r \
						-o ${OUTPUT_PATH}/${SUBJECT_ID}_rigidreg_T1strip_ \


	echo "Applying deformable registration to rigid images"
	#nonlinearly register skull stripped rigid image to skull stripped atlas (-so -> deformable syn only)
	#required for downstream transformation of subject saliency maps to atlas
       ${ANTSPATH}antsRegistrationSyNQuick.sh -d 3 \
						-f ${ATLAS_STRIPPED} \
						-m ${OUTPUT_PATH}/${SUBJECT_ID}_rigidreg_T1strip_Warped.nii.gz \
						-t so \
						-o ${OUTPUT_PATH}/${SUBJECT_ID}_deformreg_T1strip_ \


	EOF1
	chmod +x rigidreg_${SUBJECT_ID}.sh



done
