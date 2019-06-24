# Brain-fiber-classification-using-CNNs

## Description
The model for brain fiber classification is included here:
- deep CNN with focal loss + center loss + attention:
  this model was designed for the high-resolution connectome data
 
## Requirement
- [python==3.X](https://www.anaconda.com/download/)
- [pytorch==1.1 with cuda 9.0](http://pytorch.org/)
## Usage
- testing: files used to test trained CNN models using new data (.mat)
  
  For example, 
  ```
  python test_atm.py ../rDTI_CSD_sift1_tcknum_500000.mat ../models/deep_CE.model 65
  ```
Remember to change all the paths in the scripts according to your file location.
