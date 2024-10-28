# Code for "Fairness-related performance and explainability effects in deep learning models for brain image analysis"

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

## Citation
```
@article{stanley_fairness-related_2022,
	title = {Fairness-related performance and explainability effects in deep learning models for brain image analysis},
	volume = {9},
	url = {https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-9/issue-6/061102/Fairness-related-performance-and-explainability-effects-in-deep-learning-models/10.1117/1.JMI.9.6.061102.full},
	doi = {10.1117/1.JMI.9.6.061102},
	number = {6},
	urldate = {2023-10-18},
	journal = {JMI},
	author = {Stanley, Emma A. M. and Wilms, Matthias and Mouches, Pauline and Forkert, Nils D.},
	month = aug,
	year = {2022},
	pages = {061102},
}
```
