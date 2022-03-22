# Template for Notebook Tutorials in the Deep Learning 2 course

This folder represents a template for creating a new notebook tutorial for the Deep Learning 2 course at the University of Amsterdam. See `TemplateNotebook.ipynb` for instructions on how to write a tutorial. Below, we give instructions on how to create a new tutorial from scratch and add it later to this repository and the website:
* Fork this repository
* In your fork, duplicate this `template` folder and rename the new folder to the tutorial folder (we want to have one folder per notebook)
* Remove this README and the `example_image.svg` file in the new folder. If you want to add images to your notebook (which is recommended), add them to the same folder. The template gives an example of how you can include images.
* Once you finished your tutorial, give the notebook a full, new runthrough from top to bottom. The outputs of all cells will be shown on the website, so make sure everything is as you want it.
* Go to the most outer folder, and then to the file `docs/index.rst`. At the end of the file, you see a list of all notebooks. Under the tab 'Deep Learning 2', add the path to your notebook. This way it is added to the website (otherwise it will be ignored).
* It is recommended to build the website locally. You can install the necessary packages via the file `docs/requirements.txt`. Go to `docs/` and run `make html`. In the folder `_build/html/`, you can then start a simple python server, `python -m http.server`, and look at the website. Check if everything is shown as you want it.
* Finally, create a pull request from your forked repository to the `master` branch of the original uvadlc_notebooks repo. We can discuss changes to the notebook there before adding it to the repo.