# -*- coding: utf-8 -*-

'''
the files containing the callback definitions require access to the Dash app instance however if this were imported from index.py,
the initial loading of index.py would ultimately require itself to be already imported, which cannot be satisfied.
'''
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

from views.app import app
from views.elements import model_selector
from views.train_ghsom import layout as layout_train_ghsom

#from models.ghsom import GHSOM,GSOM
import numpy as np
import json

import  plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
#from views.custom_annotated_heatmap import create_annotated_heatmap as custom_heatmap

from  views.session_data import session_data
from  config.config import *

 #TESTIN DATA
#PARA NO TENER QUE RECARGAR EL DATASET EN LAS PRUEBAS
data_to_plot = [[np.nan ,np.nan ,np.nan, np.nan, np.nan ,np.nan ,np.nan, 0],
                [np.nan ,np.nan, np.nan ,5 ,np.nan, np.nan ,np.nan ,np.nan],
                [np.nan ,np.nan, np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan ,np.nan, np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan],
                [8 ,np.nan ,np.nan ,np.nan, np.nan, np.nan ,np.nan ,np.nan],
                [np.nan ,np.nan, np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan ,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ,np.nan],
                [np.nan ,np.nan, 0, np.nan, np.nan, np.nan, np.nan, 0]]

data_to_plot = [[None ,None ,None, None, None ,None ,None, 0],
                [None ,None, None ,5 ,None, None ,None ,None],
                [None ,None, None ,None, None, None, None, None],
                [None ,None, None ,None, None, None, None, None],
                [8 ,None ,None ,None, None, None ,None ,None],
                [None ,None, None ,None, None, None, None, None],
                [None ,None, None, None, None, None, None ,None],
                [None ,None, 0, None, None, None, None, 0]]
                    



#############################################################
###################  MAIN.PY CALLBACKS ######################
#############################################################


#Dropdown selector modelo
@app.callback(Output('label_selected_model', 'children'),
              Input('seleccion_modelo_som','n_clicks'),
              Input('seleccion_modelo_gsom','n_clicks'),
              Input('seleccion_modelo_ghsom','n_clicks'),
              Input('url', 'pathname') )
def dropdown_update_training_selection(n1,n2,n3,url):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]


    if (trigger_id == "seleccion_modelo_ghsom" or 'ghsom' in url ):
        return 'GHSOM'
    elif (trigger_id == "seleccion_modelo_gsom" or 'gsom' in url ):
        return 'GSOM'
    else:
        return 'SOM' 











###################  TRAINING_SELECTION.PY CALLBACKS ######################

# TODO BORRAR ESTA FUNCION
#Actualizar tabla info dataset
'''
@app.callback(Output('table_info_n_samples', 'children'),
              Output('table_info_n_features', 'children'),
              Input('session_data','modified_timestamp'),
              State('session_data', 'data'),
              prevent_initial_call=False)
def update_dataset_info_table( data, session_data):
    return session_data['n_samples'] ,session_data['n_features']

'''










   




