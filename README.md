# Changes in this fork

To use this fork of fmralign with conda package manager:
Clone this package into your local directory, then:
```
conda create -n myenvname python=3.9
cd fmralign
pip install -e .
pip install -e .[jax]
```

Changes in this fork compared to parent fmralign package:
1. surf_pairwise_alignment.py takes input images as numpy arrays instead of NIFTI
2. template_alignment.py modified to take input images as numpy arrays instead of NIFTI
3. Added template generation methods: hyperalignment, PCA method
4. Added spatial regularization (Jeganathan et al., 2024, draft)
5. Added ProMises model
6. Added SCCA regularization (Xu et al., 2012)

# fmralign

[Functional alignment for fMRI](https://parietal-inria.github.io/fmralign-docs) (functional Magnetic Resonance Imaging) data.

This light-weight Python library provides access to a range of functional alignment methods, including Procrustes and Optimal Transport.
It is compatible with and inspired by [Nilearn](http://nilearn.github.io).
Alternative implementations of these ideas can be found in the [pymvpa](http://www.pymvpa.org) or [brainiak](http://brainiak.org) packages.
The [netrep](https://github.com/ahwillia/netrep) library also offers many of the same metrics,though with a more general focus beyond fMRI data.

## Getting Started

### Installation

You can access the latest stable version of fmralign directly with the PyPi package installer:

```
pip install fmralign
```

For development or bleeding-edge features, fmralign can also be installed directly from source:

```
git clone https://github.com/Parietal-INRIA/fmralign.git
cd fmralign
pip install -e .
```

Note that if you want to use the JAX-accelerated optimal transport methods, you should also run:

```
pip install fmralign .[jax]
```

### Documentation

You can found an introduction to functional alignment, a user guide and some examples
on how to use the package at https://parietal-inria.github.io/fmralign-docs.

## License

This project is licensed under the Simplified BSD License.

## Acknowledgments

This project has received funding from the European Union’s Horizon
2020 Research and Innovation Programme under Grant Agreement No. 785907
(HBP SGA2).
This project was supported by [Digiteo](http://www.digiteo.fr).
