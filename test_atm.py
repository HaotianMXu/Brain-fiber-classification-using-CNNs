#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates 4 kinds of testing results
1) overall result (all fibers included)
    each row contains the following info:
        fiber index (length 1)
        coordinates of head and tail points (length 6)
        probs of each membership (length NCLASS)
        final membership/prediction (length 1)
        final prob/prob of predicted membership (length 1)
2) fiber index (for each membership)
    each row contains the following info:
        fiber index (length 1)
3) final prob (for each membership)
    each row contains the following info:
        final prob (length 1)
4) fiber attention map (for each membership)
    each row contains the following info:
        attention weights (length 100)
membership index starts from 0 and fiber index starts from 1
"""
import torch
from torch.autograd import Variable
import torch.utils.data as utils
import numpy as np
import gc
import sys
import scipy.io as spio
import h5py
import RESNET152_ATT_naive

"""load mat"""
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    try:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    except:
        output=dict()
        data = h5py.File(filename)
        count=data['tracks']['count'].value
        total_count=0
        for i in count:
            total_count*=10
            total_count+=int(chr(i[0]))
        track=list()
        for i in range(total_count):
            track.append(np.transpose(data[data['tracks']['data'][i].item()][:]).astype(np.float32))
        output['tracks']={}
        output['tracks']['count']=total_count
        output['tracks']['data']=track
        return output

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def mySoftmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div
"""normalize"""#110
def rescale(X_list,count):
    output=list()
    if count==1:
        output.append(X_list/110)
        return output
    for i in range(len(X_list)):
        output.append(X_list[i]/110)
    return output

def udflip(X_list,count=2):
    output=list()
    if count==1:
        output.append(np.flipud(X_list))
        return output
    for i in range(len(X_list)):
        output.append(np.flipud(X_list[i]))
    return output
def datato3d(arrays):#list of np arrays, NULL*3*100
    output=list()
    for i in arrays:
        i=np.squeeze(i,axis=1)
        i=np.transpose(i,(0,2,1))
        output.append(i)
    return output
def main(argv):
    matpath,modelpath,classnum=argv[1],argv[2],argv[3]
    """testing settings"""
    args_test_batch_size=4096
    NCLASS=int(classnum)
    """build datasets"""
    assert(matpath[-3:]=='mat')
    print('processing',matpath)
    print('start to load mat')
    mat=loadmat(matpath)
    X_test=mat['tracks']['data']
    X_test=rescale(X_test,int(mat['tracks']['count']))
    X_test=np.asarray(X_test).astype(np.float32)
    X_test=np.transpose(X_test,(0,2,1))
    del mat
    gc.collect()
    X_test_np=X_test.copy()
    X_test=torch.from_numpy(X_test)
    y_test=torch.from_numpy(np.zeros(X_test.shape[0],dtype=np.uint8))
    print('X_shape',X_test.size())
    print('data loaded!')
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    tst_set=utils.TensorDataset(X_test,y_test)
    tst_loader=utils.DataLoader(tst_set,batch_size=args_test_batch_size,shuffle=False,**kwargs)
        
    """init model"""
    model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS)
    model.cuda()

    def test():
        model.eval()
        logit=list()
        attVec=list()
        for data,lbl in tst_loader:
            with torch.no_grad():
                data = Variable(data.cuda())
            output,_,att = model(data)
            logit.append(output.data.cpu().numpy())
            attVec.append(att.data.cpu().numpy())
        return logit,attVec

    """
    test
    """
    model.load_state_dict(torch.load(modelpath))
    logit,attVec=test()
    attVec=mySoftmax(np.squeeze(np.vstack(attVec))).astype(np.float32)
    print('size of attVec',attVec.shape)
    #build output
    prob=np.exp(np.vstack(logit)).astype(np.float32)
    membership=np.argmax(prob,axis=1).reshape((-1,1)).astype(np.float32)
    maxprob=np.amax(prob,axis=1).reshape((-1,1)).astype(np.float32)
    
    output_max=np.zeros((X_test_np.shape[0],7),dtype=np.float32)
    output_max[:,0]=np.arange(1,X_test_np.shape[0]+1)
    for i in range(X_test_np.shape[0]):
        output_max[i,1:4]=X_test_np[i,:,0]
        output_max[i,4:7]=X_test_np[i,:,-1]
    #merge
    output_max=np.hstack((output_max,prob,membership,maxprob))
    """
    output
    """
    #overall result
    np.savetxt(matpath.replace('.mat','.txt'),output_max,fmt='%.4e')
    #others
    for i in range(NCLASS):
        submat=output_max[np.where(output_max[:,-2]==i)]
        #fiber index
        fiberIndex=submat[:,0].reshape((-1,1))
        np.savetxt(matpath.replace('.mat','_'+'{0:02}'.format(i)+'_fiberindex.txt'),fiberIndex,fmt='%d')
        #fiber prob
        fiberProb=submat[:,-1].reshape((-1,1))
        np.savetxt(matpath.replace('.mat','_'+'{0:02}'.format(i)+'_fiberprob.txt'),fiberProb,fmt='%.4e')
        #fiber attention map
        fiberAtm=attVec[np.where(output_max[:,-2]==i)]
        np.savetxt(matpath.replace('.mat','_'+'{0:02}'.format(i)+'_fiberatm.txt'),fiberAtm,fmt='%.4e')        
    print('results saved!')

def argvHelp():
    print('Please pass 3 variables to the file')
    print('Call this script as:')
    print('python test_atm.py /path/to/mat /path/to/model classnum')
def checkArgc(argv):
    if(len(argv)!=4):
        argvHelp()
if __name__=="__main__":
    print("\n".join(sys.argv))
    checkArgc(sys.argv)
    main(sys.argv)
