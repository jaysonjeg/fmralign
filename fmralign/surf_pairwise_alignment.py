# -*- coding: utf-8 -*-
""" 
Module for pairwise functional alignment where data are in numpy arrays (nsamples,nvertices)
Unlike pairwise_alignment.py, this module does not do any smoothing, masking, or preprocessing
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import (delayed, Parallel)
from fmralign._utils import piecewise_transform
from fmralign.pairwise_alignment import fit_one_piece
import warnings

def generate_Xi_Yi(labels, X, Y, per_parcel_kwargs, verbose):
    """ Generate source and target data X_i and Y_i for each piece i.

    Parameters
    ----------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    X: Niimg-like object
        Source data
    Y: Niimg-like object
        Target data
    per_parcel_kwargs: dict
        extra arguments, unique value for each parcel. Dictionary of keys (argument name) and values (list of values, one for each parcel) For each parcel, the part of per_parcel_kwargs that applies to that parcel will be added to alignment_kwargs
    verbose: integer, optional.
        Indicate the level of verbosity.

    Yields
    -------
    X_i: ndarray
        Source data for piece i (shape : n_samples, n_features)
    Y_i: ndarray
        Target data for piece i (shape : n_samples, n_features)
    kwargs_kth_parcel: dict
        extra arguments passed to alignment method, that are unique to each parcel

    """

    unique_labels = np.unique(labels)
    for k in range(len(unique_labels)):
        label = unique_labels[k]
        i = label == labels
        if (k + 1) % 25 == 0 and verbose > 0:
            print("Fitting parcel: " + str(k + 1) +
                  "/" + str(len(unique_labels)))
        kwargs_kth_parcel = {key: value[k] for key, value in per_parcel_kwargs.items()} #Given a dictionary with values that are lists, return a new dictionary containing only the kth element of each value
        yield X[:, i], Y[:, i], kwargs_kth_parcel


def fit_parcellation(X_, Y_, alignment_method, clustering, n_jobs, parallel_type, verbose, alignment_kwargs,per_parcel_kwargs):
    """ Create one parcellation of n_pieces and align each source and target
    data in one piece i, X_i and Y_i, using alignment method
    and learn transformation to map X to Y.

    Parameters
    ----------
    X_: Niimg-like object
        Source data
    Y_: Niimg-like object
        Target data
    alignment_method: string
        algorithm used to perform alignment between each region of X_ and Y_
    clustering: string or GiftiImage
        Surf atlas with parcels
    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
    parallel_type: string "processes" or "threads"
        Whether to parallelize with multiple processes or threads
    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed
    alignment_kwargs: dict
        extra arguments passed to alignment method. Dictionary of keys (argument name) and values (value, same for each parcel)
    per_parcel_kwargs: dict
        extra arguments, unique value for each parcel. Dictionary of keys (argument name) and values (list of values, one for each parcel) For each parcel, the part of per_parcel_kwargs that applies to that parcel is added to alignment_kwargs

    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """
    # choose indexes maybe with index_img to not
    labels = clustering

    fit = Parallel(n_jobs, prefer=parallel_type, verbose=verbose)(
        delayed(fit_one_piece)(
            X_i, Y_i, alignment_method, dict(alignment_kwargs, **kwargs_kth_parcel)
        ) for X_i, Y_i, kwargs_kth_parcel in generate_Xi_Yi(labels, X_, Y_, per_parcel_kwargs, verbose)
    )

    return labels, fit


class SurfacePairwiseAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source and target images into regions and align corresponding \
    regions independently.
    """

    def __init__(self, alignment_method, clustering, n_jobs=1, parallel_type='threads',verbose=0, alignment_kwargs={},per_parcel_kwargs={}, gamma=0):
        """
        If n_pieces > 1, decomposes the images into regions \
        and align each source/target region independantly.
        If n_bags > 1, this parcellation process is applied multiple time \
        and the resulting models are bagged.

        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes
            (imported from functional_alignment.alignment_methods)
        clustering : numpy int array (n_vertices)
            Parcellation of vertices in clusters. Each vertex is assigned a parcel number
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        parallel_type: string "processes" or "threads"
            Whether to parallelize with multiple processes or threads
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        alignment_kwargs: dict
            any extra arguments are passed to the alignment method
        per_parcel_kwargs: dict
            extra arguments, unique value for each parcel. Dictionary of keys (argument name) and values (list of values, one for each parcel) For each parcel, the part of per_parcel_kwargs that applies to that parcel will be added to alignment_kwargs
        gamma: float, optional (default = 0) range 0 to 1
            Regularization parameter. If gamma==0, then no regularization is applied. Suggest values between 0.05 and 0.2. Replaces target image with a weighted average of source and target images. 
        """
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.n_jobs = n_jobs
        self.parallel_type = parallel_type
        self.verbose = verbose
        self.alignment_kwargs=alignment_kwargs
        self.per_parcel_kwargs=per_parcel_kwargs
        self.gamma=gamma

    def fit(self, X, Y):
        """Fit data X and Y and learn transformation to map X to Y

        Parameters
        ----------
        X: source data, numpy array (n_samples, n_vertices)

        Y: target data, numpy array (n_samples, n_vertices)

        Returns
        -------
        self
        """

        from collections import Counter
        ntimepoints=X.shape[0]
        nVerticesInLargestCluster = Counter(self.clustering)[0]
        if ntimepoints < nVerticesInLargestCluster:
            warnings.warn(f'UserWarning: ntimepoints {ntimepoints} < nVerticesInLargestCluster {nVerticesInLargestCluster}')
        
        if self.gamma:
            Y = np.average([X,Y],axis=0,weights=[self.gamma,1-self.gamma])

        self.labels_, self.fit_ = fit_parcellation(
            X, Y, self.alignment_method, self.clustering, self.n_jobs, self.parallel_type, self.verbose,self.alignment_kwargs,self.per_parcel_kwargs)
        # not list here unlike pairwise

        return self

    def transform(self, X):
        """Predict data from X

        Parameters
        ----------
        X: source data, numpy array (n_samples, n_vertices)

        Returns
        -------
        X_transform: predicted data, numpy array (n_samples, n_vertices)
        """
        X_transform = piecewise_transform(
            self.labels_, self.fit_, X)

        return X_transform

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'")
