# -*- coding: utf-8 -*-
"""Creating a template for a group of subject using alignment
===================================================================

In this tutorial, we show how to better predict new contrasts for a target
subject using many source subjects corresponding contrasts. For this purpose,
we create a template to which we align the target subject.

We mostly rely on Python common packages and on nilearn to handle
functional data in a clean fashion.


To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

# Retrieving the data
# -------------------
# In this example we use the IBC dataset, which includes a large number of \
# different contrasts maps for 12 subjects.
# We download the images for subjects sub-01, 02, 04, 05, 06 and sub-07 \
# (or retrieve them if they were already downloaded).
# imgs is the list of paths to available statistical images for each subjects.
# df is a dataframe with metadata about each of them.
# mask is a binary image used to extract grey matter regions.
#

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
imgs, df, mask_img = fetch_ibc_subjects_contrasts(
    ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-06', 'sub-07'])

###############################################################################
# Defining a masker
# -----------------
# Using the mask provided, define a nilearn masker that will be used to \
# handle relevant data. For more information, visit :
# http://nilearn.github.io/manipulating_images/masker_objects.html
#

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_img).fit()

###############################################################################
# Split the data
# ---------------------------------------------
# For each subject, we will use two series of contrasts acquired during
# two independent sessions with a different phase encoding: \
# Antero-posterior(AP) or Postero-anterior(PA).
#
# We infer a template for subjects sub-01 to sub-06 for both AP and PA data.
#
# For subject sub-07, we split it in two folds, one to learn alignement, the \
# other one that we'll try to predict:
#
# * target train: AP contrasts for subject sub-07, used to learn alignment \
#   towards the template
# * target test: PA contrasts for subject sub-07, used as a ground truth to  \
#   score our predictions
#
from nilearn.image import concat_imgs

# We make a list of 4D niimgs from our list of list of files containing 3D images
template_train = []
for i in range(5):
    template_train.append(concat_imgs(imgs[i]))
target_train = df[df.subject == 'sub-07'][df.acquisition == 'ap'].path.values
# We make a single 4D Niimg from our list of 3D filenames
target_train = concat_imgs(target_train)
target_test = df[df.subject == 'sub-07'][df.acquisition == 'pa'].path.values

#############################################################################
# Compute the average of training subjects images, to be used as a baseline.
# -------------------------------------------------------------------------
# We use a function that return an image with as many contrasts as any subject
# representing the average of all train subjects.

from fmralign.template_alignment import euclidean_mean

average_subject = euclidean_mean(template_train, masker)

#############################################################################
# Create a template from the training subjects.
# ----------------------------------------------
# We define an estimator using the class TemplateAlignment:
# * We align the whole brain through 'multiple' local alignments.
# * These alignments are calculated on a parcellation of the brain in 150\
#   pieces, this parcellation creates group of functionnally similar voxels.
# * The template is created iteratively, aligning all subjects data into a
#   common space, from which the template is inferred and aligning again to this
#   new template space.

from fmralign.template_alignment import TemplateAlignment
from nilearn.image import index_img

template_estim = TemplateAlignment(
    n_pieces=150, alignment_method='ridge_cv', mask=masker)
template_estim.fit(template_train, n_iter=2)

#############################################################################
# Predict subject sub-07 PA data from the template and fitted estimator
# ----------------------------------------------------------------
# We use target_train data to fit the transform, indicating it corresponds to
# the contrasts indexed by train_index and predict from this learnt alignment
# contrasts corresponding to template test_index numbers.

# For each train subject and for the template, the AP contrasts are sorted from
# 0, to 53, and then the PA contrasts from 53 to 106.
train_index = range(53)
test_index = range(53, 106)
# We input the mapping image target_train in a list, we could have input more
# than one subject for which we'd want to predict : [train_1, train_2 ...]
prediction_from_template = template_estim.transform([target_train], train_index,
                                                    test_index)
# We can also try to predict from averaging
prediction_from_average = index_img(average_subject, test_index)

#############################################################################
# Score the prediction of test data without alignment
# ---------------------------------------------------
# To score the quality of prediction we use r2 score on each voxel \
# activation profile across contrasts. This score is 1 for a perfect prediction \
# and can get arbitrarly bad (here we clip it to -1 for bad predictions)
# We compare accuracy of predictions made from group average and from template.

import numpy as np
from sklearn.metrics import r2_score
# The PA contrasts deduced from averaging training subjects

# The baseline score represents the quality of prediction using raw data
average_score = np.maximum(r2_score(
    masker.transform(target_test), masker.transform(prediction_from_average),
    multioutput='raw_values'), -1)
# The baseline score represents the quality of prediction using aligned data
template_score = np.maximum(r2_score(
    masker.transform(target_test), masker.transform(
        prediction_from_template[0]),
    multioutput='raw_values'), - 1)

#############################################################################
# Plotting the prediction quality
# ---------------------------------------------------
#
from nilearn import plotting

baseline_display = plotting.plot_stat_map(masker.inverse_transform(
    average_score), display_mode="z", vmax=1, cut_coords=[-15, -5])
baseline_display.title("R2 score of prediction from group average")

display = plotting.plot_stat_map(
    masker.inverse_transform(
        template_score), display_mode="z", cut_coords=[-15, -5], vmax=1)
display.title("R2 score of prediction using template and alignment")
plotting.show()
#############################################################################
# We observe that creating a template and aligning a new subject to it yiels \
# better prediction of his test contrasts than just using the group average.
