# Discret2Di - Discretization for automated consistency based diagnosis

## First steps 
At the start, take a look at the instructions in the notebook file named "prepare_datasets.ipynb". 
Once you're done, you can access the preprocessed datasets for training the model and evaluating our approach. 
Also, make sure to run the "tank_generation.ipynb" file to create the simulated Three-Tank dataset.

The resources for finding the datasets are as follows: 
* BeRfiPl Dataset - Benchmark for diagnosis, reconfiguration and planning: https://github.com/j-ehrhardt/benchmark-for-diagnosis-reconf-planning
* Siemens SmartAutomation SmA proceess plant: https://github.com/thomasbierweiler/FaultsOf4-TankBatchProcess
* Secure Water Treatment Dataset: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/
* Three Tank Simulation Dataset - Simulation out of the paper "Learning Physical Concepts in CPS: A Case Study with a Three-Tank System": https://www.sciencedirect.com/science/article/pii/S2405896322004840



## Model Training of the CatVAE
Please find the model hyperparameters for our evaluation in this Repo under "Discret2Di_Appendix.pdf". <br>
The parameters can also be found in the folder /CatVAE_training_hparams/ for every single evaluation dataset. <br>
As well, you can find the tuned parameters for the Gaussian Mixture Models. 


## Evaluation 
Within the following folders, the evaluation of the specific sections of the paper can be reviewed. <br> 

**Evaluation of CatVAE** (folder path: notebooks_evaluation_CatVAE): Discretisations can be carried out in the notebooks with the pre-trained models and compared on the basis of the plots.

**Evaluation of GMMs** compared to CatVAE (folder path: notebooks_evaluation_gmm): As a baseline to the CatVAE, a GMM is trained to discretize the different datasets and compute their according likelihood. 

**Evaluation of Discret2Di** (folder path: notebooks_evaluation_discret2di): Based on the simulated data set of the three-tank model, various anomalies were simulated as described in the preprocessing step. <br>
Within the folder, the discretisations of the CatVAE can first be checked and then the notebook "discret2di_tank.ipynb" can be executed, in which the various diagnoses are carried out.
