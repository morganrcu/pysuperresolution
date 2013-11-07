print(__doc__)

from time import time

import pylab as pl
import numpy as np

import dualDictLearn
from scipy.misc import lena

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

###############################################################################
# Load Lena image and extract patches

lena = lena() / 256.0

# downsample for higher speed
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena /= 4.0
height, width = lena.shape

print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(lena, patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))

###############################################################################
# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

pl.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    pl.subplot(10, 10, i + 1)
    pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r,
              interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.suptitle('Dictionary learned from Lena patches\n' +
            'Train time %.1fs on %d patches' % (dt, len(data)),
            fontsize=16)
pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

###############################################################################
# Extract noisy patches and reconstruct them using the dictionary
transform_algorithms = [
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = lena.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    
    highDictionary=dualDictLearn.l2ls_learn_basis_dual(data.T,code.T,1)
    highDictionary=highDictionary.T
    print 'Distance: %f'% np.sum(np.abs(V-highDictionary))
    pl.figure(figsize=(4.2, 4))
    for i, comp in enumerate(highDictionary[:100]):
        pl.subplot(10, 10, i + 1)
        pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r,interpolation='nearest')
        pl.xticks(())
        pl.yticks(())
    pl.suptitle('Dictionary learned from Lena patches\n' +'Train time %.1fs on %d patches' % (dt, len(data)), fontsize=16)
    pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    pl.show()
