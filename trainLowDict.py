#!/usr/bin/python



from time import time

import tifffile as tff

import numpy as np

import pylab as pl

from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder
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
    dico = DictionaryLearning(n_components=100, alpha=1, max_iter=100,verbose=1)

    V = dico.fit(buffer).components_
    E = dico.error_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    return V,E

def getSparseCodes(dataset,Dict):
    print Dict.shape
    print dataset.shape
    coder=SparseCoder(Dict,transform_algorithm='lasso_lars')
    return coder.transform(dataset)

def trainHighDict(buffer,coeffs):
    dualDictLearn.l2ls_learn_basis_dual(buffer.T,coeffs.T,1)
    return
def superresolution():
    #numimages=3600
    numimages=3600
    lowData=readDataset('out','low',numimages,32*32*32)
    np.save('lowData',lowData)
    lowData,muLow,stdLow=standarizeDataset(lowData)

    print 'Saved....'
    print np.mean(lowData,axis=0)
    #print np.std(lowData,axis=0)
    lowDict,lowError=trainLowDict(lowData)

    lowCodes=getSparseCodes(lowData,lowDict)
    
    #print lowCodes
    np.save('lowCodes',lowCodes)
    np.save('lowDict',lowDict)
    
    highData=readDataset('out','high',numimages,64*64*64)
    highData,muHigh,stdHigh=standarizeDataset(highData)
    
    highDict=trainHighDict(highData,lowCodes)
    lowCodes=getSparseCodes(highData,highDict)
    np.save('highDict',highDict)
import dualDictLearn

    
if __name__=="__main__":
    
    superresolution()
    #highDict()
