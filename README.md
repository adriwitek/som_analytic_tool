# SOM Analytic Tool

Graphic Interactive Tool for Data Analysis and Visualization with Self Organized Maps.


## Introduction
 Self Organized Maps Algortihms are Unsupervised Artificial Neural Networks able to map  high-dimensional relationships in data into a low-dimensional, discretized space.
 Inspired in neurobiology , they change their internal structure in response to stimulus. Similar paterns are located in the same regions of space(just like human brain, different functions are located on different regions of the cortex).


## About the Tool
This is a graphic interactive tool for data analysis and visualization with self organized maps algorithms based on Dash.
  3 Algorithms are avaible at this tool:

  * Classic SOM (Self Organized Map):  Also known as Kohonen Maps.
  * GSOM(Growing Self-Organizing Map): A som that grows depending on data input.
  * GHSOM(Growing Hierarchical Self-Organizing Map): A hierarchical tree structure made with GSOMs that can grow  both vertical and horizontal depending on input data dristribution, showing the data relationships.


## Requeriments
Versions are important, since different versions on some libraries causes some problems!

| Software  | Version |
|:--------------------------------------------------------------:|:-------:|
| [Python](https://www.python.org/downloads/)                    | 3.8.3  | 
| [Dash](https://dash.plotly.com/installation)                   | 1.19.0 | 
| [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)  | 0.11.3 | 
| [Python Dateutil](https://pypi.org/project/python-dateutil/)                      |  2.8.1| 
| [Cycler](https://pypi.org/project/Cycler/)                      | 0.10.0 | 
| [Kiwisolver](https://pypi.org/project/kiwisolver/)                      |1.0.1  | 
| [Pandas](https://pypi.org/project/pandas/)                      | 1.2.4 | 
| [Networkx](https://networkx.org/)                      | 2.5| 
| [Scikit-learn](https://scikit-learn.org/stable/install.html)                      | 0.24.1 | 
| [matplotlib](https://matplotlib.org/)                          | 3.4.1   |
| [NumPy](http://www.numpy.org/)                                 | 1.20.2  | 
| [ProgressBar 2](https://pypi.org/project/progressbar2/)        | 3.37.1  | 
                 
## Installation
Recommend using an enviroment like conda for avoid packages versions problems.
```python
 pip install -r requirements.txt 
 ```
 
## Run
Open a python terminal on app's directory an then run:
```python
 python tool.py 
 ```
Then just go to http://localhost:8050/ on your Web Browser(You can click on direction on terminal showing after runnig previous command )
