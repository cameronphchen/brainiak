#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
""" Group ICA

The implementations are based on the following publications:

.. [Calhoun2009] "A review of group ICA for fMRI data and ICA for joint inference of imaging, genetic, and ERP data",
   Calhoun, Vince D., Jingyu Liu, and Tülay Adalı.
   Neuroimage, 2009

"""

# Authors: Po-Hsuan (Cameron) Chen (Princeton Neuroscience Institute) 2017

import logging

import numpy as np
import scipy
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError
from sklearn.decomposition import PCA, FastICA

__all__ = [
    "GICA"
]

logger = logging.getLogger(__name__)


class GICA(BaseEstimator, TransformerMixin):
    """Group Independent Component Analysis (GICA)

    TODO

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.


    Attributes
    ----------

    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.

    s_ : array, shape=[features, samples]
        The shared response.

    sigma_s_ : array, shape=[features, features]
        The covariance of the shared response Normal distribution.

    mu_ : list of array, element i has shape=[voxels_i]
        The voxel means over the samples for each subject.

    rho2_ : array, shape=[subjects]
        The estimated noise variance :math:`\\rho_i^2` for each subject


    Note
    ----
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        return

    def fit(self, X, y=None):

        """Compute GICA

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used

        """
        logger.info('Starting MCCA')

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError("There are not enough subjects "
                             "({0:d}) to train the model.".format(len(X)))

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features))

        # Check if all subjects have same number of TRs
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of samples between subjects"
                                 ".")
        # Run MCCA
        self.w_, self.s_ = self._gica(X)

        return self

    def transform(self, X, y=None):
        """Use the model to transform matrix to Shared Response space

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject
            note that number of voxels and samples can vary across subjects
        y : not used (as it is unsupervised learning)


        Returns
        -------
        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        s = [None] * len(X)
        for subject in range(len(X)):
            s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    
    def _objective_function(self, data, w):
        """Calculate the log-likelihood function


        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        w : list of 2D arrays, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        Returns
        -------

        loglikehood : float
            The objective function value.

        """
        return None


    def _gica(self, data):
        """inference algorithm for MCCA

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        Returns
        -------

        w : list of array, element i has shape=[voxels_i, features]
            The transforms (mappings) :math:`W_i` for each subject.
        """

        [nvoxel, nTR] = data[0].shape
        nsubjs = len(data)
        nfeature = self.features
        
        # zscore the data
        bY = np.zeros((nTR,nvoxel,nsubjs))
        for m in range(nsubjs):
            bY[:,:,m] = stats.zscore(data[m].T ,axis=0, ddof=1)
        
        # First PCA
        Fi = np.zeros((nTR,nfeature,nsubjs))
        Xi = np.zeros((nfeature,nvoxel,nsubjs))
        X_stack = np.zeros((nfeature*nsubjs,nvoxel))
        
        for m in range(nsubjs):
            U, s, VT = np.linalg.svd(bY[:,:,m], full_matrices=False)
            Fi[:,:,m] = U[:,range(nfeature)]
            Xi[:,:,m] = np.diag(s[range(nfeature)]).dot(VT[range(nfeature),:])
            X_stack[m*nfeature:(m+1)*nfeature,:] = Xi[:,:,m]
      
        # Choose N for second PCA
        U, s, VT = np.linalg.svd(X_stack, full_matrices=False)
        r = np.linalg.matrix_rank(X_stack)
        AIC  = np.zeros((r-1))
        MDL = np.zeros((r-1))
        tmp1 = 1.0
        tmp2 = 0.0
        for N in range(r-2,-1,-1):
            tmp1 = tmp1*s[N+1]
            tmp2 = tmp2+s[N+1]
            L_N = np.log(tmp1**(1/(r-1-N))/((tmp2/(r-1-N))))
            AIC[N] = -2*nvoxel*(nfeature*nsubjs-N-1)*L_N + 2*(1+(N+1)*nfeature+N/2)
            MDL[N] = -nvoxel*(nfeature*nsubjs-N-1)*L_N + 0.5*(1+(N+1)*nfeature+N/2)*np.log(nvoxel)
        
        nfeat2 = int(round(np.mean([np.argmin(AIC), np.argmin(MDL)])))+1 # N
        
        # Second PCA
        G = U[:,range(nfeat2)]
        X = np.diag(s[range(nfeat2)]).dot(VT[range(nfeat2),:]) # N-by-V
        
        # ICA
        randseed = 0
        np.random.seed(randseed) # randseed = 0
        tmp = np.mat(np.random.random((nfeat2,nfeat2)))
        
        ica = FastICA(n_components= nfeat2, max_iter=500,w_init=tmp,whiten=False,random_state=randseed)
        St = ica.fit_transform(X.T)
        S = St.T
        A = ica.mixing_

        # Partitioning
        Gi = np.zeros((nfeature,nfeat2,nsubjs))
        Si = np.zeros((nfeat2,nvoxel,nsubjs))

        for m in range(nsubjs):
            Gi[:,:,m] = G[m*nfeature:(m+1)*nfeature,:]
            Si[:,:,m] = np.linalg.pinv(Gi[:,:,m].dot(A)).dot(Xi[:,:,m])

        # Forming the factorization matrices such that Yi.T = bWi*bSi
        bW = []
        bS = []
        for m in range(nsubjs):
            bW.append(Si[:,:,m].T)
            bS.append((Fi[:,:,m].dot(Gi[:,:,m]).dot(A)).T)
        
        return bW, bS
