# -*- coding: utf-8 -*-
""" Module for functional template inference using functional alignment on Niimgs and
prediction of new subjects unseen images
"""
# Author: T. Bazeille, B. Thirion
# License: simplified BSD

import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment

def _rescaled_euclidean_mean(imgs, scale_average=False):
    """Make the Euclidian average of images

    Parameters
    ----------
    imgs: list of Niimgs
        Each img is 3D by default, but can also be 4D.
    scale_average: boolean
        If true, the returned average is scaled to have the average norm of imgs
        If false, it will usually have a smaller norm than initial average
        because noise will cancel across images

    Returns
    -------
    average_img: Niimg
        Average of imgs, with same shape as one img
    """
    average_img = np.mean(imgs, axis=0)
    scale = 1
    if scale_average:
        X_norm = 0
        for img in imgs:
            X_norm += np.linalg.norm(img)
        X_norm /= len(imgs)
        scale = X_norm / np.linalg.norm(average_img)
    average_img *= scale

    return average_img

def _align_one_image_to_target(img,alignment_method,clustering,n_jobs,target,alignment_kwargs):
    piecewise_estimator= SurfacePairwiseAlignment(alignment_method, clustering, n_jobs=n_jobs,alignment_kwargs=alignment_kwargs)
    piecewise_estimator.fit(img, target) 
    return piecewise_estimator

def _align_images_to_template(
    imgs,
    template,
    alignment_method,
    clustering,
    n_bags,
    memory,
    memory_level,
    n_jobs,
    verbose,
    alignment_kwargs,
):
    """Convenience function : for a list of images, return the list
    of estimators (PairwiseAlignment instances) aligning each of them to a
    common target, the template. All arguments are used in PairwiseAlignment
    """
    piecewise_estimators = Parallel(n_jobs=-1)(delayed(_align_one_image_to_target)(img,alignment_method,clustering,n_jobs,template,alignment_kwargs) for img in imgs)       
    aligned_imgs = [piecewise_estimators[i].transform(imgs[i]) for i in range(len(imgs))] 

    return aligned_imgs, piecewise_estimators

def _align_images_to_template_excluding_self(
    imgs,
    aligned_imgs,
    alignment_method,
    clustering,
    n_bags,
    memory,
    memory_level,
    n_jobs,
    verbose,
    alignment_kwargs,
    scale_template=None
):
    """As above, but it will align each subject to a slightly different template which is the average of all subjects excluding that subject. This requires argument scale_template which will be passed to _rescaled_euclidean_mean
    """

    get_this_template = lambda i: _rescaled_euclidean_mean([item for index,item in enumerate(aligned_imgs) if index!=i], scale_template)
    piecewise_estimators = Parallel(n_jobs=-1)(delayed(_align_one_image_to_target)(imgs[i],alignment_method,clustering,n_jobs,get_this_template(i),alignment_kwargs) for i in range(len(imgs)))       
    aligned_imgs = [piecewise_estimators[i].transform(imgs[i]) for i in range(len(imgs))] 

    return aligned_imgs, piecewise_estimators

def _create_template(
    imgs,
    n_iter,
    scale_template,
    alignment_method,
    clustering,
    n_bags,
    memory,
    memory_level,
    n_jobs,
    verbose,
    template_method=1,
    include_current_subject=True,
    alignment_kwargs={}
):
    """Create template through alternate minimization.  Compute iteratively :
    * T minimizing sum(||R_i X_i-T||) which is the mean of aligned images (RX_i)
    * align initial images to new template T
        (find transform R_i minimizing ||R_i X_i-T|| for each img X_i)


    Parameters
    ----------
    imgs: List of Niimg-like objects
       See http://nilearn.github.io/manipulating_images/input_output.html
       source data. Every img must have the same length (n_sample)
    scale_template: boolean
        If true, template is rescaled after each inference so that it keeps
        the same norm as the average of training images.
    n_iter: int
       Number of iterations in the alternate minimization. Each image is
       aligned n_iter times to the evolving template. If n_iter = 0,
       the template is simply the mean of the input images.
    template_method: int
        1: same as original fmralign code
        2: hyperalignment method, where published?, on first iteration the new template is the average of all previous aligned images
        3: hyperalignment method in https://doi.org/10.1016/j.neuron.2011.08.026 and https://doi.org/10.1371/journal.pcbi.1006120, on first iteration the new template (for next subject) is the average of the previous template and the current subject's aligned image. Needs n_iter=1.  exclude_current_subject=True for https://doi.org/10.1371/journal.pcbi, or False for https://doi.org/10.1016/j.neuron.2011.08.026
    include_current_subject: bool
        whether to align subject's to the template including themself or not
    All other arguments are the same are passed to PairwiseAlignment

    Returns
    -------
    template: list of 3D Niimgs of length (n_sample)
        Models the barycenter of input imgs
    piecewise_estimators
    """

    aligned_imgs = imgs
    for iter in range(n_iter):

        if iter==0 and template_method != 1: 
            assert(n_iter==2) #hyperalignment method needs 2 iterations after the first round
            aligned_imgs=[imgs[0]] 
            current_template = imgs[0]
            for i in range(1,len(imgs)):
                piecewise_estimator= SurfacePairwiseAlignment(alignment_method, clustering, n_jobs=n_jobs,alignment_kwargs=alignment_kwargs)
                piecewise_estimator.fit(imgs[i], current_template)
                new_img = piecewise_estimator.transform(imgs[i])
                aligned_imgs.append(new_img)
                if template_method==2: #new template is average of all previous aligned images
                    current_template = _rescaled_euclidean_mean(aligned_imgs,scale_template)
                elif template_method==3: #new template is average of previous template and latest image
                    current_template = _rescaled_euclidean_mean([current_template, new_img],scale_template)  

        if include_current_subject==True:
            template = _rescaled_euclidean_mean(aligned_imgs, scale_template)
            aligned_imgs, piecewise_estimators = _align_images_to_template(
                imgs,
                template,
                alignment_method,
                clustering,
                n_bags,
                memory,
                memory_level,
                n_jobs,
                verbose,
                alignment_kwargs
            )
        elif include_current_subject==False: 
            aligned_imgs, piecewise_estimators = _align_images_to_template_excluding_self(
                imgs,
                aligned_imgs,
                alignment_method,
                clustering,
                n_bags,
                memory,
                memory_level,
                n_jobs,
                verbose,
                alignment_kwargs, scale_template=None)
            template = _rescaled_euclidean_mean(aligned_imgs, scale_template)

    return template, piecewise_estimators




class TemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information \
    in a template, then use pairwise alignment to predict \
    new contrast for target subject.
    """

    def __init__(
        self,
        alignment_method="identity",
        clustering=None,
        scale_template=False,
        n_iter=2,
        save_template=None,
        n_bags=1,
        memory=Memory(location=None),
        memory_level=0,
        n_jobs=1,
        verbose=0,
        template_method=1,
        include_current_subject=True,
        alignment_kwargs={}
    ):
        """
        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes (imported from
            functional_alignment.alignment_methods)
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If > 1, the voxels are clustered and alignment is performed on each
            cluster applied to X and Y.
        clustering : array (int,)
            Clustering labels for each voxel
        scale_template: boolean, default False
            rescale template after each inference so that it keeps
            the same norm as the average of training images.
        n_iter: int
           number of iteration in the alternate minimization. Each img is
           aligned n_iter times to the evolving template. If n_iter = 0,
           the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.
        n_bags: int, optional (default = 1)
            If 1 : one estimator is fitted.
            If >1 number of bagged parcellations and estimators used.
        memory: instance of joblib.Memory or string (default = None)
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional (default = None)
            Rough estimator of the amount of memory used by caching.
            Higher value means more memory for caching.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        template_method: int
            1: same as original fmralign code
            2: hyperalignment method, on first iteration the new template is the average of all previous aligned images
            3: hyperalignment method, on first iteration the new template is the average of the previous template and the latest image
        alignment_kwargs: dict
            Additional keyword arguments to pass to the alignment method
        """
        self.template = None
        self.template_history = None
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.save_template = save_template
        self.n_bags = n_bags
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.template_method=template_method
        self.include_current_subject=include_current_subject
        self.alignment_kwargs = alignment_kwargs

    def fit_to_template(self,imgs):
        """
        Fit new imgs to pre-calculated template
        """
        _,self.estimators = _align_images_to_template(imgs, self.template, self.alignment_method, self.clustering, self.n_bags, self.memory, self.memory_level,self.n_jobs, self.verbose,self.alignment_kwargs)
    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of 4D Niimg-like or List of lists of 3D Niimg-like
            Source subjects data. Each element of the parent list is one subject
            data, and all must have the same length (n_samples).

        Returns
        -------
        self

        Attributes
        ----------
        self.template: 4D Niimg object
            Length : n_samples

        """
        # Assume imgs is a list (nsubjects) of arrays(nsamples,nvertices)
        assert(imgs[0].shape[1]==self.clustering.shape[0])
        self.template, self.estimators = _create_template(
            imgs,
            self.n_iter,
            self.scale_template,
            self.alignment_method,
            self.clustering,
            self.n_bags,
            self.memory,
            self.memory_level,
            self.n_jobs,
            self.verbose,
            self.template_method,
            self.include_current_subject,
            self.alignment_kwargs,
        )
        if self.save_template is not None:
            self.template.to_filename(self.save_template)


    def transform(self,X,index):
        #Transform data array X with subject 'index''s aligner to the template 
        return self.estimators[index].transform(X)


    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )
