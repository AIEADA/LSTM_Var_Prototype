[![DOI](https://zenodo.org/badge/343552676.svg)](https://zenodo.org/badge/latestdoi/343552676)

## How to use this code

Within `/Forecasting/` is a directory set up for an arbitrary learning, testing, 3D var task. The `config.yaml` file sets up the specifics of the task and the `/data/` folder has the training and testing data sets. Within `config.yaml`, we need to specify paths to the different data sets, whether we want to (re)-train, test or perform 3DVar (note training must always be performed first for the latter two) and specify some hyperparameters related to the training, testing (such as number of windows in and out etc. Once the configuration is prepared in the yaml file - run 
```python
python source/main.py
```
and 
```python
python source/comparisons.py
```
for performing your task. Note that a copy of the `config.yaml` will be saved with your results (you also have to specify your path to the results). 
