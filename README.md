# Brain-fiber-classification-using-CNNs

## Description
Three diffrent kinds of models for brain fiber classification are included here:

- shallow CNN with focal loss: 

  this model was designed for 15-class ESM data and the result was published in our ISBI paper

- deep CNN with focal loss + center loss:

  this model was designed for 65-class TMI data and result was reported in our TMI paper

- deep CNN with focal loss + center loss + attention:
  this model was designed for the high-resolution connectome data
 
## Requirement
- [python==3.X](https://www.anaconda.com/download/)
- [pytorch==0.3.1 with GPU support](http://pytorch.org/)
- [scikit-learn==0.19.1](http://scikit-learn.org/)
## Usage
Each method contains three folds: 

- preprocessing: files used to generate training and validation data from .mat data

  More specifically, for **shallow CNN with focal loss** and **deep CNN with focal loss + center loss**, first run **subjectSplit_s7.py** to randomly split subjects and then run **mat2pkl4sys7.py** to extract fiber data from .mat files and build datasets for CNN.
  
  For **deep CNN with focal loss + center loss + attention**, simply run **mat2pkl.py**.
  
  Make sure each fiber is represented by 100 points with 3d coordinates.

- training: files used to train CNN models 

  Run **main.py**, and the model with best performance will be saved. You may want to choose your own hyperparameters.

- testing: files used to test trained CNN models using new data (.mat)

  Given all the testing fibers in a single mat file, you need this command:
  ```
  python test_XXX.py /path/to/mat/file /path/to/trained/model classnum
  ```
  
  For example, 
  ```
  python test_deep.py ../rDTI_CSD_sift1_tcknum_500000.mat ../models/deep_CE.model 65
  ```
Remember to change all the paths in the scripts based on your file location.



