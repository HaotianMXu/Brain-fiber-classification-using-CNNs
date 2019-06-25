# Brain-fiber-classification-using-CNNs

## Description
This code is part of the implementation of our TMI paper Objective Detection of Eloquent Axonal Pathways to Minimize Postoperative Deficits in Pediatric Epilepsy Surgery using Diffusion Tractography and Convolutional Neural Networks. The model for brain fiber classification is included here:
- deep CNN with focal loss + center loss + attention.
 
## The scripts are verfied on Ubuntu with
- [python==3.X](https://www.anaconda.com/download/)
- [pytorch==1.1 with cuda 9.0](http://pytorch.org/)
## Usage
- test: an example .mat file can be found [here](https://drive.google.com/file/d/1jsoRmF9J_bSxKBPV3FFLmVxe-amX3bnJ/view?usp=sharing). The example output folder is provided.
  
  For example, 
  ```
  python /path/to/test_atm.py ./example.mat ./DCNN_CL_ATT.model 65
  ```
Remember to change all the paths in the scripts according to your file location. The class indices are shown as below:
![Image of Table II](https://github.com/HaotianMXu/Brain-fiber-classification-using-CNNs/blob/master/index_to_class.PNG)

