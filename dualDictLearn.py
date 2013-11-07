
import numpy as np
import scipy
import scipy.optimize

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

np.set_printoptions(threshold='nan')

def fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt):

    # Local Variables: SSt_inv, c, g, XSt, f, H, M, L, temp, SSt, X, trXXt, dual_lambda
    # Function calls: trace, diag, inv, nargout, length, zeros, fobj_basis_dual, sum, size
    #% Compute the objective function value at x
    L = XSt.shape[0]

    M = dual_lambda.shape[0]
    SSt_inv = np.linalg.inv((SSt+np.diag(dual_lambda)))
    #% trXXt = sum(sum(X.^2));
    if L > M:
        #% (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
        f = -np.trace(np.dot(SSt_inv, np.dot(XSt.T, XSt)))+trXXt-c*np.sum(dual_lambda)
    else:
        #% (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
        f = -np.trace(np.dot(np.dot(XSt, SSt_inv), XSt.T))+trXXt-c*np.sum(dual_lambda)
    #print 'F'
    print f
    f = -f
    return f
def fobj_basis_dual_g(dual_lambda, SSt, XSt, X, c, trXXt):
    
    SSt_inv = np.linalg.inv((SSt+np.diag(dual_lambda)))
    
    #% trXXt = sum(sum(X.^2));
        
        #% fun called with two output arguments
    #% Gradient of the function evaluated at x

    temp = np.dot(XSt, SSt_inv)
    g = np.sum(temp**2,axis=0)-c
#    print 'G'
    g=-g
    return g

def fobj_basis_dual_H(dual_lambda, SSt, XSt, X, c, trXXt):

    # Local Variables: SSt_inv, c, g, XSt, f, H, M, L, temp, SSt, X, trXXt, dual_lambda
    # Function calls: trace, diag, inv, nargout, length, zeros, fobj_basis_dual, sum, size
    #% Compute the objective function value at x
   
    SSt_inv = np.linalg.inv((SSt+np.diag(dual_lambda)))

    temp = np.dot(XSt, SSt_inv)

    
    H = -2.*np.dot(temp.T, temp)*SSt_inv
    H = -H
#    print 'Hessian'
#    print H
    
    return H

def l2ls_learn_basis_dual(X, S, l2norm):

    # Local Variables: fobjective_dual, lb, XSt, fval_opt, fval, B_dual, dual_lambda, B, SSt, M, L, exitflag, N, Binit, l2norm, x, X, c, S, fobjective, output, trXXt, options

    # Function calls: rand, fmincon, diag, sum, optimset, toc, Bt, abs, exist, l2ls_learn_basis_dual, fobj_basis_dual, zeros, tic, size

    #% Learning basis using Lagrange dual (with basis normalization)
    #%
    #% This code solves the following problem:
    #% 
    #%    minimize_B   0.5*||X - B*S||^2
    #%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
    #% 
    #% The detail of the algorithm is described in the following paper:
    #% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
    #% Advances in Neural Information Processing Systems (NIPS) 19, 2007
    #%
    #% Written by Honglak Lee <hllee@cs.stanford.edu>
    #% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng
    print X.shape
    print S.shape

    L=X.shape[0] #L = matcompat.size(X, 1.)  
    N= X.shape[1] #N = matcompat.size(X, 2.)
    M= S.shape[0] #M = matcompat.size(S, 1.)
    #tic
    print L
    print N
    print M

    SSt = np.dot(S, S.T)
    XSt = np.dot(X, S.T)
    
    print SSt.shape
    print XSt.shape
    # if exist('Binit', 'var'):
    #     dual_lambda = np.diag((linalg.solve(Binit, XSt)-SSt))
    #else:
    dual_lambda = 10.*np.abs(np.random.rand(M))
    #% any arbitrary initialization should be ok.
    
    c = l2norm**2.
    trXXt = np.sum(X**2.)
    #lb = np.zeros(matcompat.size(dual_lambda))
    #lb = np.zeros(dual_lambda.shape)
    #    options = optimset('GradObj', 'on', 'Hessian', 'on')
    #%  options = optimset('GradObj','on', 'Hessian','on', 'TolFun', 1e-7);
    lb = [ (0,None) for i in range(M)]
    solution = scipy.optimize.minimize(fobj_basis_dual, dual_lambda,args=(SSt,XSt,X,c,trXXt),bounds=lb, method='TNC',jac=fobj_basis_dual_g, hess=fobj_basis_dual_H,options={'disp': True})
    x=solution.x
    fval=solution.fun
    print x
#    [x, fval, exitflag, output] = fmincon(lambda x: fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, np.array([]), np.array([]), np.array([]), np.array([]), lb, np.array([]), np.array([]), options)
    #% output.iterations
    fval_opt = np.dot(np.dot(-0.5, N), fval)
    dual_lambda = x
    Bt = np.dot(np.linalg.inv(SSt+np.diag(dual_lambda)), XSt.T);
    
    B_dual = Bt.T
    fobjective_dual = fval_opt
    B = B_dual
    print B.shape
    fobjective = fobjective_dual
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    return [B]
import trainLowDict
def highDict():
    lowCodes=np.load('lowCodes.npy')
    numimages=360
    highData=trainLowDict.readDataset('out','high',numimages,64*64*64)
    highData,muHigh,stdHigh=trainLowDict.standarizeDataset(highData)
    l2ls_learn_basis_dual(highData.T,lowCodes.T,1)
if __name__=="__main__":
    import sys
    #superresolution()
    highDict()