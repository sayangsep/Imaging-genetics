# Imaging-genetics
A Generative Discriminative Framework that Integrates Imaging, Genetic, andDiagnosis Data into Coupled Low Dimensional Space

Abstract:
We propose a novel optimization framework that integrates imaging and genetics data for simultaneous biomarker identification and disease classification. 
The generative component of our model uses a dictionary learning framework to project the imaging and genetic data into a shared low dimensional space. 
We have coupled both the data modalities by tying the  linear projection coefficients to the same latent space. 
The discriminative component of our model uses logistic regression on the projection vectors for disease diagnosis. 
This prediction task implicitly guides our framework to find interpretable biomarkers that are substantially different between a healthy and disease population. 
We exploit the interconnectedness of different brain regions by incorporating a graph regularization penalty into the joint objective function.
We also use a group sparsity penalty to find a representative set of genetic basis vectors that span a low dimensional space where subjects are easily separable between patients and controls.
We have evaluated our model on a population study of schizophrenia that includes two task fMRI paradigms and single nucleotide polymorphism (SNP) data. 


File:
cross_validation_script_libd_nback.m, and cross_validation_script_libd_sdmt.m files contains the script to run a cross validation analysis on two differet datasets
NBack and SDMT obtained from Lieber Institute of Brain Imaging (LIBD). Both the data set contains brain activation maps of ROIs as defined by the brain connectome atlas.

NBack Data:

data/nback 

This folder contains files of brain activation maps and SNPs data for each subject. N0-Back file contains brain activation of the N0-Back task and N2-Back file contains 
brain activation of the N2-Back task. In our analysis we used the contrast maps (N2-N0).

SDMT Data:

data/sdmt 

This folder contains files of brain activation maps and SNPs data for each subject for the SDMT task. Crosshair file contains brain activation when the subjects are shown crosshair
and Aversive file contains brain activation when the subjects are shown aversive images. In our analysis we used the contrast maps (Aversive-Crosshair).
