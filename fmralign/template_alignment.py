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

'''
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
'''


def _align_images_to_target(source_imgs,target_img,clustering,alignment_method,alignment_kwargs,gamma=0):
    """
    Parallelize calling _align_one_image for all source_imgs
    """
    aligners= Parallel(n_jobs=-1,prefer='processes')(delayed(_align_one_image)(source_img,target_img,clustering,alignment_method,alignment_kwargs,n_jobs=1,parallel_type='threads',gamma=gamma) for source_img in source_imgs)
    return aligners
def _align_one_image(source_img,target_img,clustering,alignment_method,alignment_kwargs,n_jobs=1,parallel_type='threads',gamma=0):
    aligner= SurfacePairwiseAlignment(alignment_method, clustering, n_jobs=n_jobs, parallel_type=parallel_type,alignment_kwargs=alignment_kwargs,gamma=gamma)
    aligner.fit(source_img, target_img)
    return aligner
def _align_one_image_without_self(img,template,level_1_aligned_img,n_imgs,normalizer_template,clustering,alignment_method,alignment_kwargs,n_jobs=1,parallel_type='threads',gamma=0):
    #similar to _align_one_image, except it also subtracts the subject's aligned image from the template before aligning the other images to it
    template = (template * n_imgs - level_1_aligned_img) / (n_imgs - 1)
    template = normalizer_template(template)
    aligner = _align_one_image(img,template,clustering,alignment_method,alignment_kwargs,n_jobs=n_jobs,parallel_type=parallel_type,gamma=gamma)
    return aligner
def zscore(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)
def normalize_to_unit_norm(X):
    return X/np.linalg.norm(X)
   

class TemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information \
    in a template, then use pairwise alignment to predict \
    new contrast for target subject.
    """

    def __init__(self,alignment_method="identity",clustering=None,alignment_kwargs={}):
        """
        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes (imported from
            functional_alignment.alignment_methods)
        clustering: array: shape (nvertices,), dtype int
            Clustering labels for each vertex
        alignment_kwargs: dict
            Additional keyword arguments to pass to the alignment method
        """
        self.template = None
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.alignment_kwargs = alignment_kwargs

    def make_template(self,imgs,n_iter,do_level_1,level1_equal_weight,normalize_imgs,normalize_template,remove_self,gamma=0):
        """
        Make template image from a list of images. Combines elements of code from fmralign package and pyMVPA2 package, in particular using some similar naming as pyMVPA2. This function does 'level 1' (optional) and 'level 2' of pyMVPA2. Level 1 involves iteratively aligning images to an evolving template. Level 2 simultaneously aligns all images to a single template
        For standard hyperalignment: do_level_1=True, normalize_imgs='zscore', normalize_template='zscore', remove_self=True, level1_equal_weight=False
        For standard fmralign: do_level_1=False, normalize_imgs=None, normalize_template=None,remove_self=False
            This is the GPA method: ref Gower, J. C. , & Dijksterhuis, G. B. (2004). Procrustes problems (Vol. 30). Oxford University Press on Demand
        Parameters
        ----------
        imgs: list(nsubjects) of arrays (nsamples,nvertices)
            Source data
        n_iter: int
            number of iterations at level 2. Set to zero just to use mean of level 1 images as the template
        do_level_1: bool
            whether to do level 1. 
        level1_equal_weight: bool
            in level 1, weight each subject equally in the template. 
        normalize_imgs: 'zscore', 'rescale', or None
            To normalize each aligned image before taking the mean to make template. 'rescale' scales each image to have unit matrix Frobenius norm. 'zscore' zscores each column, so that each image contributes near-equally to each vertex. For hyperalignment, pick 'zscore'. 'rescale' might have problems if the original matrix norm is too big for dtype e.g. float16
        normalize_template: 'zscore', 'rescale', or None
            To normalize the template at each iteration. 
        remove_self: bool
            in level 2, subtract that subject's aligned image from the template before aligning the other images to it, so that the subject is not included in the template that they are aligned to. 
        gamma: float [0 to 1]
            regularization parameter for surf_pairwise_alignment
        """


        normalize_dict = {'zscore':zscore, 'rescale':normalize_to_unit_norm, None:lambda X: X}
        normalizer_imgs = normalize_dict[normalize_imgs]
        normalizer_template = normalize_dict[normalize_template]
        if (normalize_template=='rescale' or normalize_imgs=='rescale') and (imgs[0].dtype==np.float16):
            imgs = [i.astype(np.float32) for i in imgs] #rescaling might have problems if the original matrix norm is too big for dtype e.g. float16

        #Level 1
        if do_level_1: 
            template = normalizer_template(imgs[0]) #initial template is first subject's image
            aligned_imgs = [imgs[0]] #consider renaming this variable as aligned_imgs !!!
            for i in range(1,len(imgs)):
                aligner = _align_one_image(imgs[i],template,self.clustering,self.alignment_method,self.alignment_kwargs,n_jobs=-1,parallel_type='threads',gamma=gamma)
                new_img = aligner.transform(imgs[i]) #image aligned to template
                new_img = normalizer_imgs(new_img)
                aligned_imgs.append(new_img) #slow step, maybe initialise as empty numpy array of objects? !!!
                if level1_equal_weight: 
                    template = np.average([template, new_img], weights=(i, 1.0), axis=0) #think they have it backwards in pymvpa
                else: 
                    template = np.mean([template,new_img],axis=0) #in this scenario, earlier images have lesser contribution to the template
                template = normalizer_template(template)
        else:
            aligned_imgs = [normalizer_imgs(i) for i in imgs]
        
        #Level 2
        template = np.mean(aligned_imgs,axis=0) #initial template is average of aligned images
        for iter in range(n_iter):
            if remove_self: 
                n_imgs = len(imgs)
                aligners = Parallel(n_jobs=-1,prefer='processes')(delayed(_align_one_image_without_self)(img,template,aligned_img,n_imgs,normalizer_template,self.clustering,self.alignment_method,self.alignment_kwargs,n_jobs=1,parallel_type='threads',gamma=gamma) for img,aligned_img in zip(imgs,aligned_imgs))    
            else:      
                template = normalizer_template(template)                
                aligners = _align_images_to_target(imgs,template,self.clustering,self.alignment_method,self.alignment_kwargs,gamma)
            aligned_imgs = [aligners[i].transform(imgs[i]) for i in range(len(imgs))] 
            aligned_imgs = [normalizer_imgs(i) for i in aligned_imgs]
            template = np.mean(aligned_imgs,axis=0)
        self.template = normalizer_template(template)


    def fit_to_template(self,imgs,gamma=0):
        """
        Fit new imgs to pre-calculated template
        """
        self.estimators = _align_images_to_target(imgs,self.template,self.clustering,self.alignment_method,self.alignment_kwargs,gamma) 

    def transform(self,X,index):
        #Transform data array X with subject 'index''s aligner to the template 
        return self.estimators[index].transform(X)


    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )
