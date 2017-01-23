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
""" Multiset CCA

The implementations are based on the following publications:

.. [Kettenring1971] "Canonical analysis of several sets of variables",
   Jon R Kettenring
   Biometrika, 1971
   

.. [Li2016] "Joint blind source separation by multiset canonical correlation analysis",
   Yi-Ou Li
   IEEE Transactions on Signal Processing, 2009
"""

# Authors: Po-Hsuan (Cameron) Chen (Princeton Neuroscience Institute) 2017

import logging

import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError

__all__ = [
    "MCCA"
]

logger = logging.getLogger(__name__)


class MCCA(BaseEstimator, TransformerMixin):
    """Multiset Canonical Correlation Analysis (MCCA)

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

    def fit(self, X, y=None, objfunc='MAXVAR'):

        """Compute MCCA

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used

        objfunc : 
            (1) the sum of correlations method (SUMCOR)
            (2) the maximum variance method (MAXVAR)
            (3) the sum of squared correlations method (SSQCOR)
            (4) the minimum variance method (MINVAR)
            (5) the generalized variance method (GENVAR)
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

        if objfunc not in ['SUMCOR', 'MAXVAR', 'SSQCOR', 'MINVAR', 'GENVAR']:
            raise ValueError("Incorrect objective function")

        # Run MCCA
        self.w_, self.s_ = self._mcca(X, objfunc)

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
        obj = 0
        for i in range(subjects):
            for j in range(i+1, subjects):
                obj += np.trace(w[i].T.dot(data[i]).dot(data[j].T.dot(w[j])))
        return obj


    def _mcca(self, data, objfunc):
        """inference algorithm for MCCA

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        objfunc : ['SUMCOR', 'MAXVAR', 'SSQCOR', 'MINVAR', 'GENVAR']

        Returns
        -------

        w : list of array, element i has shape=[voxels_i, features]
            The transforms (mappings) :math:`W_i` for each subject.
        """

        samples = data[0].shape[1]
        subjects = len(data)

        np.random.seed(self.rand_seed)

        w = []
        A = []
        newdata = []
        subjects = len(data)
        voxels = np.empty(subjects, dtype=int)

        for subject in range(subjects):
            perturbation = np.zeros(data[subject].shape)
            np.fill_diagonal(perturbation, 0.001)
            u_subject, s_subject, v_subject = np.linalg.svd(data[subject] + perturbation, full_matrices=False)
            #print u_subject.shape
            #print np.linalg.pinv(np.diag(s_subject)).shape
            #print u_subject.shape
            A.append(u_subject.dot(np.linalg.pinv(np.diag(s_subject)).dot(u_subject.T)))
            newdata.append(A[subject].dot(data[subject]))

        # Set Wi to a random orthogonal voxels by features matrix
        for subject in range(subjects):
            voxels[subject] = data[subject].shape[0]
            rnd_matrix = np.random.random((voxels[subject], self.features))
            q, r = np.linalg.qr(rnd_matrix)
            w.append(q)

        # Main loop of the algorithm (run
        for iteration in range(self.n_iter):
            logger.info('Iteration %d' % (iteration + 1))

            shared_response = np.zeros((self.features, samples))
            for subject in range(subjects):
                shared_response += w[subject].T.dot(newdata[subject])
            shared_response /= subjects

            for subject in range(subjects):
                u, _, v = np.linalg.svd(newdata[subject].dot(shared_response.T), full_matrices=False)
                w[subject] = u.dot(v)

        for subject in range(subjects):
            w[subject] = np.linalg.pinv(A[subject]).dot(w[subject])

        if logger.isEnabledFor(logging.INFO):
            # Calculate the current objective function value
            objective = self._objective_function(data, w)
            logger.info('Objective function %f' % objective)

        return w, shared_response
