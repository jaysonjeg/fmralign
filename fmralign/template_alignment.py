# -*- coding: utf-8 -*-
""" Module for functional template inference using functional alignment on Niimgs and
prediction of new subjects unseen images
"""
# Author: T. Bazeille, B. Thirion
# License: simplified BSD

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FastICA, IncrementalPCA
from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment

def _yield_imgs_one_parcel(clustering,imgs):
    """
    Called by make_lowdim_template.
    Given list (nsubjects) of arrays (ntimepoints,nvertices), yields a list (nsubjects) of arrays (ntimepoints,nvertices_in_single_parcel)
    """
    unique_labels=np.unique(clustering)
    for k in range(len(unique_labels)):
        label = unique_labels[k]
        indices = clustering == label
        imgs_one_parcel = [img[:,indices] for img in imgs]   
        yield imgs_one_parcel

def _do_dim_reduction(imgs_one_parcel,method):
    """
    Called by make_lowdim_template.
    Given a list of identically sized 2D arrays, concatenate along horizontal axis, then do PCA on that axis and retain enough components so that the dimensionality reduced version has same shape as any of the original 2D arrays
    """
    imgs_one_parcel_concat = np.hstack(imgs_one_parcel) #concatenate all subjects' data across vertices
    n_components = imgs_one_parcel[0].shape[1] #retain same no of components as no of vertices in this parcel
    if method=='pca':
        dimreduce = PCA(n_components=n_components, whiten=False,random_state=0)
    elif method=='increm_pca':
        dimreduce = IncrementalPCA(n_components=n_components,whiten=False)
    if method=='ica':
        dimreduce = FastICA(n_components=n_components,max_iter=100000) #default max_iter 200
    newimgs = dimreduce.fit_transform(imgs_one_parcel_concat)
    return newimgs

def _combine_parcelwise_imgs(clustering,imgs,shape):
    """
    Called by make_lowdim_template
    Combines time series data for each parcel separately into whole-brain time series
    clustering: array (nvertices) of ints
    imgs: list (nparcels) of arrays (ntimepoints,nvertices_in_each_parcel)
    shape: tuple
        shape of output array
    """
    result=np.zeros(shape,dtype=imgs[0].dtype)
    unique_labels=np.unique(clustering)
    for k in range(len(unique_labels)):
        label = unique_labels[k]
        indices = clustering == label
        result[:,indices] = imgs[k]
    return result   

def _align_images_to_target(source_imgs,target_img,clustering,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags,gamma=0):
    """
    Parallelize calling _align_one_image for all source_imgs
    """
    aligners= Parallel(n_jobs=-1,prefer='processes')(delayed(_align_one_image)(source_img,target_img,clustering,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags,n_jobs=1,parallel_type='threads',gamma=gamma) for source_img in source_imgs)
    return aligners
def _align_one_image(source_img,target_img,clustering,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags,n_jobs=1,parallel_type='threads',gamma=0):
    aligner= SurfacePairwiseAlignment(alignment_method, clustering, n_bags=n_bags, n_jobs=n_jobs, parallel_type=parallel_type,alignment_kwargs=alignment_kwargs,per_parcel_kwargs=per_parcel_kwargs,gamma=gamma)
    aligner.fit(source_img, target_img)
    return aligner
def _align_one_image_without_self(img,template,level_1_aligned_img,n_imgs,normalizer_template,clustering,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags, n_jobs=1,parallel_type='threads',gamma=0):
    #similar to _align_one_image, except it also subtracts the subject's aligned image from the template before aligning the other images to it
    template = (template * n_imgs - level_1_aligned_img) / (n_imgs - 1)
    template = normalizer_template(template)
    aligner = _align_one_image(img,template,clustering,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags, n_jobs=n_jobs,parallel_type=parallel_type,gamma=gamma)
    return aligner
def zscore(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)
def normalize(X,target_norm):
    #print(f'target is {target_norm}, X norm is {np.linalg.norm(X)}')
    return target_norm*(X/np.linalg.norm(X))
   
class TemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information \
    in a template, then use pairwise alignment to predict \
    new contrast for target subject.
    """

    def __init__(self,alignment_method="identity",clustering=None,alignment_kwargs={},per_parcel_kwargs={}):
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
        per_parcel_kwargs: dict
            extra arguments, unique value for each parcel. Dictionary of keys (argument name) and values (list of values, one for each parcel) For each parcel, the part of per_parcel_kwargs that applies to that parcel will be added to alignment_kwargs
        """
        self.template = None
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.alignment_kwargs = alignment_kwargs
        self.per_parcel_kwargs = per_parcel_kwargs


    def make_template(self,imgs,n_bags=1,n_iter=1,do_level_1=False,level1_equal_weight=False,normalize_imgs=None,normalize_template=None,remove_self=False,gamma=0):
        """
        Make template image from a list of images. Combines elements of code from fmralign package and pyMVPA2 package, in particular using some similar naming as pyMVPA2. This function does 'level 1' (optional) and 'level 2' of pyMVPA2. Level 1 involves iteratively aligning images to an evolving template. Level 2 simultaneously aligns all images to a single template
        For standard hyperalignment: n_iter=1,do_level_1=True, normalize_imgs='zscore', normalize_template='zscore', remove_self=True, level1_equal_weight=False
        For standard fmralign: n_iter=2,do_level_1=False, normalize_imgs='rescale', normalize_template='rescale',remove_self=False
            This is the GPA method: ref Gower, J. C. , & Dijksterhuis, G. B. (2004). Procrustes problems (Vol. 30). Oxford University Press on Demand
        Parameters
        ----------
        imgs: list(nsubjects) of arrays (nsamples,nvertices)
            Source data
        n_bags: int, default 1
            Number of bootstrap resamples in each pairwise alignment
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
            regularization parameter for surf_pairwise_alignment. If non-zero, make sure normalize_imgs and normalize_template are not None
        """

        if (normalize_template=='rescale' or normalize_imgs=='rescale') and (imgs[0].dtype==np.float16):
            imgs = [i.astype(np.float32) for i in imgs] #rescaling might have problems if the original matrix norm is too big for dtype e.g. float16
            avg_norm = np.mean([np.linalg.norm(i) for i in imgs])
        normalize_to_avg_norm = lambda img: normalize(img,avg_norm)
        normalize_dict = {'zscore':zscore, 'rescale':normalize_to_avg_norm, None:lambda X: X}
        normalizer_imgs = normalize_dict[normalize_imgs]
        normalizer_template = normalize_dict[normalize_template]

        #Level 1
        if do_level_1: 
            template = normalizer_template(imgs[0]) #initial template is first subject's image
            aligned_imgs = [imgs[0]]
            for i in range(1,len(imgs)):
                aligner = _align_one_image(imgs[i],template,self.clustering,self.alignment_method,self.alignment_kwargs,self.per_parcel_kwargs,n_bags,n_jobs=-1,parallel_type='processes',gamma=gamma)
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
                aligners = Parallel(n_jobs=-1,prefer='processes')(delayed(_align_one_image_without_self)(img,template,aligned_img,n_imgs,normalizer_template,self.clustering,self.alignment_method,self.alignment_kwargs,self.per_parcel_kwargs,n_bags,n_jobs=1,parallel_type='threads',gamma=gamma) for img,aligned_img in zip(imgs,aligned_imgs))    
            else:      
                template = normalizer_template(template)                
                aligners = _align_images_to_target(imgs,template,self.clustering,self.alignment_method,self.alignment_kwargs,self.per_parcel_kwargs,n_bags,gamma)
            aligned_imgs = [aligners[i].transform(imgs[i]) for i in range(len(imgs))] 
            aligned_imgs = [normalizer_imgs(i) for i in aligned_imgs]
            template = np.mean(aligned_imgs,axis=0)
        self.template = normalizer_template(template)

    def make_lowdim_template(self,imgs,clustering,n_bags=1,method='pca'):
        """
        Make a template time series using dimensionality reduction.
        Step 1: For each parcel, stack subjects' data across vertices to form a (time,nsubjects*nvertices_in_parcel) matrix, then do dimensionality reduction (PCA or ICA). Components will be linear combinations of some vertices in some subjects. The parcel's template is given by the time series of the first n components, where n is the number of vertices in that parcel originally. 
        Step 2: Combine parcel-specific templates into whole-brain template.
        Step 3: Within each parcel, uses Procrustes alignment to rotate the template, so as to maximize overlap with the group mean time series in anatomical space.

        INPUTS:
        imgs: list (nsubjects) of arrays (ntimepoints,nvertices)
            Brain image data
        clustering: array (nvertices) of ints
            Parcel labels
        n_bags: int, default 1
            Number of bootstrap resamples in each pairwise alignment
        method: string
            'pca', 'ica', or 'increm_pca'

        RETURNS:
        lowdim_template_rotated: array (ntimepoints,nvertices)
        """
        
        imgs_parcelwise_transformed = Parallel(n_jobs=-1)(delayed(_do_dim_reduction)(imgs_one_parcel,method) for imgs_one_parcel in _yield_imgs_one_parcel(clustering,imgs)) #Step 1
        lowdim_template = _combine_parcelwise_imgs(clustering,imgs_parcelwise_transformed,imgs[0].shape) #Step 2

        #Step 3
        aligner = SurfacePairwiseAlignment(alignment_method='scaled_orthogonal',clustering=clustering,alignment_kwargs ={'scaling':True},parallel_type='processes',n_bags=n_bags,n_jobs=-1) 
        aligner.fit( np.tile(lowdim_template,(len(imgs),1)) , np.vstack(imgs) )
        template = zscore(aligner.transform(lowdim_template))
        self.template = template

    def fit_to_template(self,imgs,n_bags=1,gamma=0):
        """
        Fit new imgs to pre-calculated template
        n_bags: int, default 1
            Number of bootstrap resamples in each pairwise alignment
        """
        if gamma!=0: #as a heuristic, check that the mean columnwise variance of first image and template image are similar. If not, then taking weighted average of an image and the template will not be appropriate
            assert(0.8 < imgs[0].var(axis=0).mean() < 1.2)
            assert(0.8 < self.template.var(axis=0).mean() < 1.2)
        self.estimators = _align_images_to_target(imgs,self.template,self.clustering,self.alignment_method,self.alignment_kwargs,self.per_parcel_kwargs,n_bags,gamma) 

    def transform(self,X,index):
        #Transform data array X with subject 'index''s aligner to the template 
        return self.estimators[index].transform(X)


    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )
