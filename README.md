UvA Deep Learning Tutorials
===========================

*Note: To look at the notebooks in a nicer format, visit our RTD website: https://uvadlc-notebooks.readthedocs.io/en/latest/*

*Course website*: https://uvadlc.github.io/<br>
*Course edition*: Fall 2024 (Oct. 28 - Dec. 20) - Being kept up to date</br>
*Recordings*: [YouTube Playlist](<https://www.youtube.com/playlist?list=PLdlPlO1QhMiAkedeu0aJixfkknLRxk1nA>)</br>
*Author*: Phillip Lippe

For this year's course edition, we created a series of Jupyter notebooks that are designed to help you understanding the "theory" from the lectures by seeing corresponding implementations.
We will visit various topics such as optimization techniques, transformers, graph neural networks, and more (for a full list, see below).
The notebooks are there to help you understand the material and teach you details of the **PyTorch** framework, including **PyTorch Lightning**.
Further, we provide one-to-one translations of the notebooks to **JAX+Flax** as alternative framework.

The notebooks are presented in the first hour of every group tutorial session.
During the tutorial sessions, we will present the content and explain the implementation of the notebooks.
You can decide yourself whether you just want to look at the filled notebook, want to try it yourself, or code along during the practical session.
The notebooks are not directly part of any mandatory assignments on which you would be graded or similarly.
However, we encourage you to get familiar with the notebooks and experiment or extend them yourself.
Further, the content presented will be relevant for the graded assignment and exam.

The tutorials have been integrated as official tutorials of PyTorch Lightning.
Thus, you can also view them in [their documentation](https://pytorch-lightning.readthedocs.io/en/latest/).

How to run the notebooks
------------------------

On this website, you will find the notebooks exported into a HTML format so that you can read them from whatever device you prefer. However, we suggest that you also give them a try and run them yourself. There are three main ways of running the notebooks we recommend:

- **Locally on CPU**: All notebooks are stored on the github repository that also builds this website. You can find them here: https://github.com/phlippe/uvadlc_notebooks/tree/master/docs/tutorial_notebooks. The notebooks are designed so that you can execute them on common laptops without the necessity of a GPU. We provide pretrained models that are automatically downloaded when running the notebooks, or can manually be downloaded from this [Google Drive](https://drive.google.com/drive/folders/1SevzqrkhHPAifKEHo-gi7J-dVxifvs4c?usp=sharing). The required disk space for the pretrained models and datasets is less than 1GB. To ensure that you have all the right python packages installed, we provide a conda environment in the [same repository](https://github.com/phlippe/uvadlc_notebooks/blob/master/) (choose the CPU or GPU version depending on your system).

- **Google Colab**: If you prefer to run the notebooks on a different platform than your own computer, or want to experiment with GPU support, we recommend using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true). Each notebook on this documentation website has a badge with a link to open it on Google Colab. Remember to enable GPU support before running the notebook (`Runtime -> Change runtime type`). Each notebook can be executed independently, and doesn't require you to connect your Google Drive or similar. However, when closing the session, changes might be lost if you don't save it to your local computer or have copied the notebook to your Google Drive beforehand.

- **Snellius cluster**: If you want to train your own (larger) neural networks based on the notebooks, you can make use of the Snellius cluster. However, this is only suggested if you really want to train a new model, and use the other two options to go through the discussion and analysis of the models. Snellius might not allow you with your student account to run Jupyter notebooks directly on the gpu_shared partition. Instead, you can first convert the notebooks to a script using `jupyter nbconvert --to script ...ipynb`, and then start a job on Snellius for running the script. A few advices when running on Snellius:
   - Disable the tqdm statements in the notebook. Otherwise your slurm output file might overflow and be several MB large. In PyTorch Lightning, you can do this by setting `progress_bar_refresh_rate=0` in the trainer.
   - Comment out the matplotlib plotting statements, or change :code:`plt.show()` to `plt.savefig(...)`.

Tutorial-Lecture alignment
--------------------------

We will discuss 7 of the tutorials in the course, spread across lectures to cover something from every area. You can align the tutorials with the lectures based on their topics. The list of tutorials is:

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

This is the first time we present these tutorials during the Deep Learning course. As with any other project, small bugs and issues are expected. We appreciate any feedback from students, whether it is about a spelling mistake, implementation bug, or suggestions for improvements/additions to the notebooks. Please use the following [link](https://docs.google.com/forms/d/e/1FAIpQLSeIhwrFSHlDSWGAgCN-RcTKm7Sn7P6bxzIyzIGge6xId1K8DQ/viewform?usp=sf_link) to submit feedback, or feel free to reach out to me directly per mail (p dot lippe at uva dot nl), or grab me during any TA session.

If you find the tutorials helpful and would like to cite them, you can use the following bibtex:
```bibtex
@misc{lippe2024uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = 2024,
   howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
}
```
