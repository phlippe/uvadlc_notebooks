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

How to run the notebooks
------------------------

On this website, you will find the notebooks exported into a HTML format so that you can read them from whatever device you prefer. However, we suggest that you also give them a try and run them yourself. There are three main ways of running the notebooks we recommend:

- **Locally on CPU**: All notebooks are stored on the github repository that also builds this website. You can find them here: https://github.com/phlippe/uvadlc_notebooks/tree/master/docs/tutorial_notebooks. The notebooks are designed that you can execute them on common laptops without the necessity of a GPU. We provide pretrained models that are automatically downloaded when running the notebooks, or can manually be downloaoded from this `Google Drive <https://drive.google.com/drive/folders/1SevzqrkhHPAifKEHo-gi7J-dVxifvs4c?usp=sharing>`_. The required disk space for the pretrained models and datasets is less than 1GB. To ensure that you have all the right python packages installed, we provide a conda environment in the `same repository <https://github.com/phlippe/uvadlc_notebooks/blob/master/dl2020_environment.yml>`_. 

- **Google Colab**: If you prefer to run the notebooks on a different platform than your own computer, or want to experiment with GPU support, we recommend using `Google Colab <https://colab.research.google.com/notebooks/intro.ipynb#recent=true>`_. Each notebook on this documentation website has a badge with a link to open it on Google Colab. Remember to enable GPU support before running the notebook (:code:`Runtime -> Change runtime type`). Each notebook can be executed independently, and doesn't require you to connect your Google Drive or similar. However, when closing the session, changes might be lost if you don't save it to your local computer or have copied the notebook to your Google Drive beforehand.

- **Lisa cluster**: If you want to train your own (larger) neural networks based on the notebooks, you can make use of the Lisa cluster. However, this is only suggested if you really want to train a new model, and use the other two options to go through the discussion and analysis of the models. Lisa might not allow you with your student account to run jupyter notebooks directly on the gpu_shared partition. Instead, you can first convert the notebooks to a script using :code:`jupyter nbconvert --to script ...ipynb`, and then start a job on Lisa for running the script. A few advices when running on Lisa:
   
   - Disable the tqdm statements in the notebook. Otherwise your slurm output file might overflow and be several MB large. In PyTorch Lightning, you can do this by setting :code:`progress_bar_refresh_rate=0` in the trainer.
   - Comment out the matplotlib plotting statements, or change :code:`plt.show()` to :code:`plt.savefig(...)`.

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

This is the first time we present these tutorials during the Deep Learning course. As with any other project, small bugs and issues are expected. We appreciate any feedback from students, whether it is about a spelling mistake, implementation bug, or suggestions for improvements/additions to the notebooks. Please use the following `link <https://docs.google.com/forms/d/e/1FAIpQLSeIhwrFSHlDSWGAgCN-RcTKm7Sn7P6bxzIyzIGge6xId1K8DQ/viewform?usp=sf_link>`_ to submit feedback, or feel free to reach out to me directly per mail (p dot lippe at uva dot nl), or grab me during any TA session.

Current progress
----------------

Not all tutorials have been finished yet, and some are still in the progress of being created. Below you can find an overview of the progress status. Each notebook has also a badge indicating its status (In progress, First version, Finished).


**Tutorials finished**

- Tutorial 1: Working with the Lisa cluster
- Tutorial 2: Introduction to PyTorch
- Tutorial 3: Activation functions
- Tutorial 4: Optimization and Initialization
- Tutorial 5: Inception, ResNet and DenseNet
- Tutorial 9: Autoencoders
- Tutorial 11: Normalizing Flows on image modeling
- Tutorial 12: Autoregressive Image Modeling

**Tutorials in work**

- Tutorial 7: Graph Neural Networks
- Tutorial 8: Deep Energy Models 
- Tutorial 10: Adversarial attacks 
- Tutorial 13: Bayesian Deep Learning (Christina)

**Tutorials left todo**

- Tutorial 6: RNNs and Attention
- Tutorial 14: Deep Dynamics


.. toctree::
   :caption: Jupyter notebooks
   :maxdepth: 2

   tutorial_notebooks/tutorial1/Lisa_Cluster
   tutorial_notebooks/tutorial2/Introduction_to_PyTorch
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