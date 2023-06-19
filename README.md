# Data Augmentation for Multivariate Times Series Classification

## Description

- Implementation and evaluation of augmentation methods on a multivariate time series dataset. Augmented datasets are then used to train the ROCKET classifer and results are analysed.

## Repository

- dataset_analysis.ipynb: initial exploritory analysis of datast.

- aug_1_reverse.ipynb: code for reversing multivariate time series.

- aug_2_windw_warp.ipynb: code for window warping augmentation method.

- aug_3_dtw-int.ipynb: code for Dynamic Time Warping and interpolation augmentation method.

- evaluation.ipynb: code that runs tests and analysis on augmented datasets.

- augmentation.py: module that contains code to run each of the augmentation methods.

- mpdatasets.py: code that loads the dataset into a pandas DataFrame.

- rocket.py: code to run the ROCKET classifier.

- requirements.txt: python libraries required to run this project.

- OpenPose_MP: Military Press dataset

- Images: stores the matplotlib charts created in the project