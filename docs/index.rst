.. notebook_test documentation master file, created by
   sphinx-quickstart on Sat Jul 25 11:56:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the UvA Deep Learning Tutorials!
===========================================

| *Course website*: https://uvadlc.github.io/
| *Course edition*: Fall 2020 (Oct. 26 - Dec. 14)

For this year's course edition, we created a series of Jupyter notebooks that are designed to help you understanding the "theory" from the lectures by seeing corresponding implementations. 
We will visit various topics such as optimization techniques, graph neural networks and adversarial attacks (for a full list, see below).
The notebooks are there to help you understand the material and teach you details of the PyTorch framework, including PyTorch Lightning. 

The notebooks are presented in the first 30-45 minutes of each practical session.
We provide two versions for each notebook: a filled one, and one with blanks for some code parts. 
During the tutorial sessions, we will present the content and do "live coding" by filling in the blanks with you.
You can decide yourself rather you just want to look at the filled notebook, want to try it yourself, or code along during the practical session.
We do not have any mandatory assignments on which you would be graded or similarly. 
However, we encourage you to get familiar with the notebooks and experiment or extend them yourself.

Prerequisites
-------------

- Install environment: _link_
- Saved models: _link_

How to run the notebooks
~~~~~~~~~~~~~~~~~~~~~~~~

- Locally on CPU (no GPU required, we provide pre-trained models)
- Google Colab
- Lisa (eventually convert to script first: :code:`jupyter nbconvert --to script ...ipynb`)

Tutorial-Lecture alignment
--------------------------

We will discuss 14 tutorials in total, each focusing on a different aspect of Deep Learning. The tutorials are spread across lectures, and we tried to cover something from every area. You can align the tutorials with the lectures as follows:

- Lecture 1: Introduction to Deep Learning 
   - Tutorial 1: Working with the Lisa cluster
   - Tutorial 2: Introduction to PyTorch
- Lecture 2: Modular Learning
   - Tutorial 3: Activation functions
- Lecture 3: Deep Learning Optimizations 
   - Tutorial 4: Optimization and Initialization
- Lecture 4: Convolutional Neural Networks
- Lecture 5: Modern ConvNets
   - Tutorial 5: Inception, ResNet and DenseNet
- Lecture 6: Recurrent Neural Networks
   - Tutorial 6: To be announced...
- Lecture 7: Graph Neural Networks
   - Tutorial 7: To be announced...
- Lecture 8: Deep Generative Models
   - Tutorial 8: Deep Energy Models
- Lecture 9: Deep Variational Inference
   - Tutorial 9: Autoencoder
- Lecture 10: Generative Adversarial Networks
   - Tutorial 10: Adversarial Attacks
- Lecture 11: Advanced Generative Models
   - Tutorial 11: Normalizing Flows
   - Tutorial 12: Autoregressive Image Modeling
- Lecture 12: Deep Stochastic Models
- Lecture 13: Bayesian Deep Learning
   - Tutorial 13: Bayesian Deep Learning
- Lecture 14: Deep Dynamics
   - Tutorial 14: To be announced...

Feedback and Contribution
-------------------------

This is the first time we present these tutorials during the Deep Learning course. As with any other project, small bugs and issues can be included. We appreciate any feedback from students, whether it is about a spelling mistake, implementation bug, or suggestions for improvements/additions to the notebooks. Please use the following link [ADD LINK] to submit feedback, or feel free to reach out to me directly per mail (p dot lippe at uva dot nl), or grab me during any TA session.

Current progress
----------------

**Tutorials finished**

- Tutorial 3.1: Activation functions
- Tutorial 4.1: Optimization and Initialization (conclusion and last part)
- Tutorial 5.1: Inception, ResNet and DenseNet
- Tutorial 9.1: Autoencoders
- Tutorial 11.2: Normalizing Flows on image modeling
- Tutorial 12.1: Autoregressive Image Modeling (code needs commenting)

**Tutorials in work**

- Tutorial 1.1: Working with the Lisa cluster
- Tutorial 8.1: Deep Energy Models (coding + experiments + commenting)
- Tutorial 10.1: Adversarial attacks (draw story and commenting)
- Tutorial 13.1: Bayesian Deep Learning (Christina)

**Tutorials left todo (noone worked on yet)**

- Tutorial 2.1+2.2: Introduction to PyTorch
- Tutorial 6.1: RNNs and Attention
- Tutorial 7.1: Graph Neural Networks
- Tutorial 14.1: Deep Dynamics (eventually drop if more time is needed for answering questions)


.. toctree::
   :caption: Jupyter notebooks
   :maxdepth: 2

   tutorial_notebooks/tutorial1/Lisa_Cluster
   tutorial2
   tutorial_notebooks/tutorial3/Activation_Functions
   tutorial_notebooks/tutorial4/Optimization_and_Initialization
   tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet
   tutorial6
   tutorial7
   tutorial8
   tutorial_notebooks/tutorial9/AE_CIFAR10
   tutorial10
   tutorial_notebooks/tutorial11/NF_image_modeling
   tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling
   tutorial13
   tutorial14