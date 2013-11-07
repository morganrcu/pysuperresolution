#!/usr/bin/python



from time import time

import tifffile as tff

import numpy as np

import pylab as pl

from sklearn.decomposition import DictionaryLearning
np.set_printoptions(threshold='nan')

def readDataset(dir,prefix,numfiles,imgSize):
    buffer=np.zeros((numfiles,imgSize),np.float32)
    for i in range(numfiles) :
        fname = '%s/%s-%d.tif'%(dir,prefix,i)
        data=tff.imread(fname)
        data=np.reshape(data,(data.size,-1))
        buffer[i,:] = data.flatten()
    return buffer

def standarizeDataset(buffer):
    mu= np.mean(buffer, axis=0)
    buffer -= mu
    std= np.std(buffer, axis=0);
    std[std==0]=1
    buffer /= std
    return buffer,mu,std

def trainLowDict(buffer):
    print('Learning the dictionary...')
    t0 = time()
    dico = DictionaryLearning(n_components=100, alpha=1, max_iter=50,verbose=1)

    V = dico.fit(buffer).components_
    E = dico.error_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    print E
    return V,E

def getSparseCodes(dataset,Dict):
    return
def trainHighDict(buffer,coefs):
    return
def superresolution():
    numimages=360
    lowData=readDataset('out','low',numimages,32*32*32)
    lowData,muLow,stdLow=standarizeDataset(lowData)
    lowDict,lowError=trainLowDict(lowData)
    lowCodes=getSparseCodes(lowData,lowDict)
    
    highData=readDataset('out','high',numimages,64*64*64)
    highData,muHigh,stdHigh=standarizeDataset(highData)
    
    highDict,highError=trainHighDict(highData)

    

if __name__=="__main__":
    import sys
    superresolution()
    
