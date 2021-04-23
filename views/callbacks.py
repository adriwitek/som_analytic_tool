# -*- coding: utf-8 -*-




from views.app import app
from views.train_ghsom import layout as layout_train_ghsom


import  plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff




'''
the files containing the callback definitions require access to the Dash app instance however if this were imported from index.py,
the initial loading of index.py would ultimately require itself to be already imported, which cannot be satisfied.
'''














   




