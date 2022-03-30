# MDSC 689.03: Advanced Medical Image Processing Final Project

## Code
### /preprocessing
* **rigid_deform_register.sh**: bash script for applying rigid + deformable registrations of each subject's T1w MRI to the NIHPD atlas 
* **mni_to_nihpd.sh**: bash script for registration of MNI ICBM152 atlas to NIHPD atlas 
* **cerebra_to_nihpd.sh**: bash script for applying MNI -> NIHPD transformation to CerebrA parcellation atlas 


### /model_saliency
* **ABCD_Rigid_CNN_sexClassification_SFCN_allsubjects_multiStrat_ValSet_byFold_norm.py**: python code for training CNN
* **Generate_saliency_ABCD_sexclassifcation_CC.py**: python code for generating individual saliency maps
* **GenerateRegisteredAverageMap_ABCDsexclass.py**: python code for combining individual saliency maps into average maps for each subgroup

### /analysis
* **ABCD_SFCN_sexclassficiation_analysis.py**: python code for analyzing performance differences between sex, race, and SES
* **segment_threshold_saliency.py**: python code for thresholding and binarzing each subgroup's average saliency maps using VTK
* **overlap_measures_saliency.py**: python code for computing Dice coefficients between each subgroup's average saliency map using simpleITK
* **saliency_ROIs.py**: python code for determining saliency scores in regions defined by CerebrA atlas with VTK
