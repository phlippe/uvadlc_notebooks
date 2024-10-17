.. notebook_test documentation master file, created by
   sphinx-quickstart on Sat Jul 25 11:56:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the UvA Deep Learning Tutorials!
===========================================

| *Course website*: https://uvadlc.github.io/
| *Course edition*: DL1 - Fall 2024, DL2 - Spring 2023, Being kept up to date
| *Repository*: https://github.com/phlippe/uvadlc_notebooks
| *Recordings*: `YouTube Playlist <https://www.youtube.com/playlist?list=PLdlPlO1QhMiAkedeu0aJixfkknLRxk1nA>`_
| *Author*: Phillip Lippe

.. note::
   Interested in learning JAX? We have recently started translating the notebooks from PyTorch to JAX+Flax. Check out our new notebooks in the tab *Deep Learning 1 (JAX+Flax)* to learn how you can speedup the model training with JAX!


For this year's course edition, we created a series of Jupyter notebooks that are designed to help you understanding the "theory" from the lectures by seeing corresponding implementations.
We will visit various topics such as optimization techniques, transformers, graph neural networks, and more (for a full list, see below).
The notebooks are there to help you understand the material and teach you details of the PyTorch framework, including PyTorch Lightning.
Further, we provide one-to-one translations of the notebooks to JAX+Flax as alternative framework.

The notebooks are presented in the first hour of every group tutorial session.
During the tutorial sessions, we will present the content and explain the implementation of the notebooks.
You can decide yourself whether you just want to look at the filled notebook, want to try it yourself, or code along during the practical session.
The notebooks are not directly part of any mandatory assignments on which you would be graded or similarly.
However, we encourage you to get familiar with the notebooks and experiment or extend them yourself.
Further, the content presented will be relevant for the graded assignment and exam.

The tutorials have been integrated as official tutorials of PyTorch Lightning.
Thus, you can also view them in `their documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_.

Schedule (Deep Learning 1, edition 2024)
----------------------------------------

+------------------------------------------+---------------------------------------------------------------+
| **Date**                                 | **Notebook**                                                  |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 29. October 2024, 09:00-10:00   | Tutorial 2: Introduction to PyTorch                           |
+------------------------------------------+---------------------------------------------------------------+
| Monday, 4. November 2024, 15:00-16:00    | Tutorial 3: Activation functions                              |
+------------------------------------------+---------------------------------------------------------------+
| Monday, 11. November 2024, 15:00-16:00   | Tutorial 4: Optimization and Initialization                   |
+------------------------------------------+---------------------------------------------------------------+
| Monday, 18. November 2024, 15:00-16:00   | Tutorial 5: Inception, ResNet and DenseNet                    |
+------------------------------------------+---------------------------------------------------------------+
| Monday, 25. November 2024, 15:00-16:00   | Tutorial 6: Transformers and Multi-Head Attention             |
+------------------------------------------+---------------------------------------------------------------+
| Monday, 2. December 2024, 15:00-16:00    | Tutorial 7: Graph Neural Networks                             |
+------------------------------------------+---------------------------------------------------------------+
| Monday, 9. December 2024, 15:00-16:00    | Tutorial 17: Self-Supervised Contrastive Learning with SimCLR |
+------------------------------------------+---------------------------------------------------------------+

How to run the notebooks
------------------------

On this website, you will find the notebooks exported into a HTML format so that you can read them from whatever device you prefer.
However, we suggest that you also give them a try and run them yourself. There are three main ways of running the notebooks we recommend:

- **Locally on CPU**: All notebooks are stored on the github repository that also builds this website. You can find them here: https://github.com/phlippe/uvadlc_notebooks/tree/master/docs/tutorial_notebooks. The notebooks are designed so that you can execute them on common laptops without the necessity of a GPU. We provide pretrained models that are automatically downloaded when running the notebooks, or can manually be downloaded from this `Google Drive <https://drive.google.com/drive/folders/1SevzqrkhHPAifKEHo-gi7J-dVxifvs4c?usp=sharing>`_. The required disk space for the pretrained models and datasets is less than 1GB. To ensure that you have all the right python packages installed, we provide a conda environment in the `same repository <https://github.com/uvadlc/uvadlc_practicals_2020/blob/master/environment.yml>`_ (choose the CPU or GPU version depending on your system).

- **Google Colab**: If you prefer to run the notebooks on a different platform than your own computer, or want to experiment with GPU support, we recommend using `Google Colab <https://colab.research.google.com/notebooks/intro.ipynb#recent=true>`_. Each notebook on this documentation website has a badge with a link to open it on Google Colab. Remember to enable GPU support before running the notebook (:code:`Runtime -> Change runtime type`). Each notebook can be executed independently, and doesn't require you to connect your Google Drive or similar. However, when closing the session, changes might be lost if you don't save it to your local computer or have copied the notebook to your Google Drive beforehand.

- **Snellius cluster**: If you want to train your own (larger) neural networks based on the notebooks, you can make use of the Snellius cluster. However, this is only suggested if you really want to train a new model, and use the other two options to go through the discussion and analysis of the models. Snellius might not allow you with your student account to run Jupyter notebooks directly on the gpu partition. Instead, you can first convert the notebooks to a script using :code:`jupyter nbconvert --to script ...ipynb`, and then start a job on Snellius for running the script. A few advices when running on Snellius:

   - Disable the tqdm statements in the notebook. Otherwise your slurm output file might overflow and be several MB large. In PyTorch Lightning, you can do this by setting :code:`enable_progress_bar=False` in the trainer.
   - Comment out the matplotlib plotting statements, or change :code:`plt.show()` to :code:`plt.savefig(...)`.

Tutorial-Lecture alignment
--------------------------

We will discuss 7 of the tutorials in the course, spread across lectures to cover something from every area. You can align the tutorials with the lectures based on their topics. The list of tutorials in the Deep Learning 1 course is:

- Guide 1: Working with the Snellius cluster
- Tutorial 2: Introduction to PyTorch
- Tutorial 3: Activation functions
- Tutorial 4: Optimization and Initialization
- Tutorial 5: Inception, ResNet and DenseNet
- Tutorial 6: Transformers and Multi-Head Attention
- Tutorial 7: Graph Neural Networks
- Tutorial 8: Deep Energy Models
- Tutorial 9: Autoencoders
- Tutorial 10: Adversarial attacks
- Tutorial 11: Normalizing Flows on image modeling
- Tutorial 12: Autoregressive Image Modeling
- Tutorial 15: Vision Transformers
- Tutorial 16: Meta Learning - Learning to Learn
- Tutorial 17: Self-Supervised Contrastive Learning with SimCLR


Feedback, Questions or Contributions
------------------------------------

This is the first time we present these tutorials during the Deep Learning course. As with any other project, small bugs and issues are expected. We appreciate any feedback from students, whether it is about a spelling mistake, implementation bug, or suggestions for improvements/additions to the notebooks. Please use the following `link <https://forms.gle/kENuNvcCq3LzQWDA8>`_ to submit feedback, or feel free to reach out to me directly per mail (p dot lippe at uva dot nl), or grab me during any TA session.

If you find the tutorials helpful and would like to cite them, you can use the following bibtex::

   @misc{lippe2024uvadlc,
      title        = {{UvA Deep Learning Tutorials}},
      author       = {Phillip Lippe},
      year         = 2024,
      howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
   }


.. toctree::
   :caption: Guides
   :maxdepth: 2

   tutorial_notebooks/tutorial1/Lisa_Cluster
   tutorial_notebooks/guide2/Research_Projects
   tutorial_notebooks/guide3/Debugging_PyTorch
   tutorial_notebooks/guide4/Research_Projects_with_JAX

.. toctree::
   :caption: Training Models at Scale
   :maxdepth: 2

   tutorial_notebooks/scaling/JAX/overview
   tutorial_notebooks/scaling/JAX/single_gpu_techniques
   tutorial_notebooks/scaling/JAX/single_gpu_transformer
   tutorial_notebooks/scaling/JAX/data_parallel_intro
   tutorial_notebooks/scaling/JAX/data_parallel_fsdp
   tutorial_notebooks/scaling/JAX/pipeline_parallel_simple
   tutorial_notebooks/scaling/JAX/pipeline_parallel_looping
   tutorial_notebooks/scaling/JAX/tensor_parallel_simple
   tutorial_notebooks/scaling/JAX/tensor_parallel_async
   tutorial_notebooks/scaling/JAX/tensor_parallel_transformer
   tutorial_notebooks/scaling/JAX/3d_parallelism

.. toctree::
   :caption: Deep Learning 1 (PyTorch)
   :maxdepth: 2

   tutorial_notebooks/tutorial2/Introduction_to_PyTorch
   tutorial_notebooks/tutorial3/Activation_Functions
   tutorial_notebooks/tutorial4/Optimization_and_Initialization
   tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet
   tutorial_notebooks/tutorial6/Transformers_and_MHAttention
   tutorial_notebooks/tutorial7/GNN_overview
   tutorial_notebooks/tutorial8/Deep_Energy_Models
   tutorial_notebooks/tutorial9/AE_CIFAR10
   tutorial_notebooks/tutorial10/Adversarial_Attacks
   tutorial_notebooks/tutorial11/NF_image_modeling
   tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling
   tutorial_notebooks/tutorial15/Vision_Transformer
   tutorial_notebooks/tutorial16/Meta_Learning
   tutorial_notebooks/tutorial17/SimCLR

.. toctree::
   :caption: Deep Learning 1 (JAX+Flax)
   :maxdepth: 2

   tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX
   tutorial_notebooks/JAX/tutorial3/Activation_Functions
   tutorial_notebooks/JAX/tutorial4/Optimization_and_Initialization
   tutorial_notebooks/JAX/tutorial5/Inception_ResNet_DenseNet
   tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention
   tutorial_notebooks/JAX/tutorial7/GNN_overview
   tutorial_notebooks/JAX/tutorial9/AE_CIFAR10
   tutorial_notebooks/JAX/tutorial11/NF_image_modeling
   tutorial_notebooks/JAX/tutorial12/Autoregressive_Image_Modeling
   tutorial_notebooks/JAX/tutorial15/Vision_Transformer
   tutorial_notebooks/JAX/tutorial17/SimCLR

.. toctree::
   :caption: Deep Learning 2
   :maxdepth: 2

   tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions
   tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns
   tutorial_notebooks/DL2/deep_probabilistic_models_I/tutorial_1.ipynb
   tutorial_notebooks/DL2/deep_probabilistic_models_II/tutorial_2a.ipynb
   tutorial_notebooks/DL2/deep_probabilistic_models_II/tutorial_2b.ipynb
   tutorial_notebooks/DL2/Advanced_Generative_Models/Normalizing_flows/advancednormflow.ipynb
   tutorial_notebooks/DL2/High-performant_DL/hyperparameter_search/hpdlhyperparam.ipynb
   tutorial_notebooks/DL2/High-performant_DL/Multi_GPU/hpdlmultigpu.ipynb
   tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.ipynb
   tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut2_student_with_answers.ipynb
   tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_1.ipynb
   tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_2.ipynb
   tutorial_notebooks/DL2/Dynamical_systems/dynamical_systems_neural_odes.ipynb
   tutorial_notebooks/DL2/sampling/introduction.ipynb
   tutorial_notebooks/DL2/sampling/subsets.ipynb
   tutorial_notebooks/DL2/sampling/permutations.ipynb
   tutorial_notebooks/DL2/sampling/graphs.ipynb
   tutorial_notebooks/DL2/Causality_and_CRL/citris-tutorial.ipynb

