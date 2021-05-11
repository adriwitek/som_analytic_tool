# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements
from models.som import minisom
import numpy as np
import plotly.graph_objects as go

from math import sqrt,ceil
from  views.session_data import session_data
from  config.config import *
import time

def train_som_view():

    '''
        A rule of thumb to set the size of the grid for a dimensionalityreduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.
    '''
    n_samples = session_data.get_train_data_n_samples()
    grid_recommended_size = ceil(  sqrt(5*sqrt(n_samples))  )

    # Formulario SOM    
    formulario_som = dbc.ListGroupItem([
                html.H4('Parameter Selection',className="card-title"  ),

                html.Div(style={'textAlign': 'center'},children=[
                        html.Div(
                            style={'display': 'inline-block', 'text-align': 'left'},
                            children=[

                                html.H5(children='Vertical Grid Size:'),
                                dcc.Input(id="tam_eje_vertical", type="number", value=grid_recommended_size,step=1,min=1),

                                html.H5(children='Horizontal Grid Size'),
                                dcc.Input(id="tam_eje_horizontal", type="number", value=grid_recommended_size,step=1,min=1),

                                html.H5(children='Learning Rate'),
                                dcc.Input(id="tasa_aprendizaje_som", type="number", value="0.5",step=0.0001,min=0,max=5),


                                html.H5(children='Neighborhood Function'),
                                dcc.Dropdown(
                                    id='dropdown_vecindad',
                                    options=[
                                        {'label': 'Gaussian', 'value': 'gaussian'},
                                        {'label': 'Mexican Hat', 'value': 'mexican_hat'},
                                        {'label': 'Bubble', 'value': 'bubble'},
                                        {'label': 'Triangle', 'value': 'triangle'}
                                    ],
                                    value='gaussian',
                                    searchable=False
                                    #style={'width': '50%'}
                                ),


                                html.H5(children='Map Topology'),
                                dcc.Dropdown(
                                    id='dropdown_topology',
                                    options=[
                                        {'label': 'Rectangular', 'value': 'rectangular'},
                                        {'label': 'Hexagonal', 'value': 'hexagonal'}
                                    ],
                                    value='rectangular',
                                    searchable=False
                                    #style={'width': '35%'}
                                ),


                                html.H5(children='Distance Function'),
                                dcc.Dropdown(
                                    id='dropdown_distance',
                                    options=[
                                        {'label': 'Euclidean', 'value': 'euclidean'},
                                        {'label': 'Cosine', 'value': 'cosine'},
                                        {'label': 'Manhattan', 'value': 'manhattan'},
                                        {'label': 'Chebyshev', 'value': 'chebyshev'}
                                    ],
                                    value='euclidean',
                                    searchable=False
                                    #style={'width': '35%'}
                                ),


                                html.H5(children='Gaussian Sigma'),
                                dcc.Input(id="sigma", type="number", value="1.5",step=0.000001,min=0,max=10),


                                html.H5(children='Max Iterations'),
                                dcc.Input(id="iteracciones", type="number", value="1000",step=1,min=1),

                                html.H5(children='Weights Initialization'),
                                dcc.Dropdown(
                                    id='dropdown_inicializacion_pesos',
                                    options=[
                                        {'label': 'PCA: Análisis de Componentes Principales ', 'value': 'pca'},
                                        {'label': 'Random', 'value': 'random'},
                                        {'label': 'No Weight Initialization ', 'value': 'no_init'}
                                    ],
                                    value='pca',
                                    searchable=False
                                    #style={'width': '45%'}
                                ),


                                    html.H5(children='Seed'),
                                    html.Div( 
                                            [dbc.Checklist(
                                                options=[{"label": "Select Seed", "value": 1}],
                                                value=[],
                                                id="check_semilla_som")]
                                    ),
                                    html.Div( id= 'div_semilla_som',
                                                children = [dcc.Input(id="seed_som", type="number", value="0",step=1,min=0, max=(2**32 - 1))],
                                                style={ "visibility": "hidden",'display':'none'}
                                    ),   


                                html.Hr(),

                                html.Div(children=[
                                    dbc.Button("Train", id="train_button_som",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary")]
                                    #,dbc.Spinner(id='spinner_training',color="primary",fullscreen=False)],
                                    #    style={'textAlign': 'center'}
                                ),
                            #Todo mirar esto
                            html.H6(id='som_entrenado')

                    ])
                ])
    ])




    ###############################   LAYOUT     ##############################
    layout = html.Div(children=[

        elements.navigation_bar,
        formulario_som,
    ])


    return layout





# Checklist seleccionar semilla
@app.callback(
    Output('div_semilla_som','style'),
    Input("check_semilla_som", "value"),
    prevent_initial_call=True
    )
def select_seed(check):

    if(check):
        return {  'display': 'block'}
    else:
        return { "visibility": "hidden",'display':'none'}




#Habilitar boton train som
@app.callback(Output('train_button_som','disabled'),
              Input('tam_eje_vertical', 'value'),
              Input('tam_eje_horizontal', 'value'),
              Input('tasa_aprendizaje_som', 'value'),
              Input('dropdown_vecindad', 'value'),
              Input('dropdown_topology', 'value'),
              Input('dropdown_distance', 'value'),
              Input('sigma', 'value'),
              Input('iteracciones', 'value'),
              Input('dropdown_inicializacion_pesos','value'),
              Input('seed_som','value'),
              Input("check_semilla_som", "value"),
            )
def enable_train_som_button(tam_eje_vertical,tam_eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,
                            sigma,iteracciones,dropdown_inicializacion_pesos,  seed, check_semilla):

    params  = [tam_eje_vertical,tam_eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,
                                    sigma,iteracciones,dropdown_inicializacion_pesos]

    if(check_semilla):
        params.append(seed)

    if all(i is not None for i in params):
        return False
    else:
        return True





@app.callback(Output('som_entrenado', 'children'),
              Input('train_button_som', 'n_clicks'),
              State('tam_eje_vertical', 'value'),
              State('tam_eje_horizontal', 'value'),
              State('tasa_aprendizaje_som', 'value'),
              State('dropdown_vecindad', 'value'),
              State('dropdown_topology', 'value'),
              State('dropdown_distance', 'value'),
              State('sigma', 'value'),
              State('iteracciones', 'value'),
              State('dropdown_inicializacion_pesos','value'),
              State('seed_som','value'),
              State("check_semilla_som", "value"),
              prevent_initial_call=True )
def train_som(n_clicks,eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,pesos_init, semilla, check_semilla):


    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)
    iteracciones = int(iteracciones)
    if(check_semilla):
        seed = int(semilla)
        check = 1
    else:
        seed = None
        check = 0

    data = session_data.get_train_data()

    start = time.time()
    session_data.start_timer()
    
    # TRAINING
    

    #ojo en numpy: array[ejevertical][ejehorizontal] ,al contratio que en plotly
    session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,distance,sigma,iteracciones, pesos_init,check,seed)

    som = minisom.MiniSom(x=eje_vertical, y=eje_horizontal, input_len=data.shape[1], sigma=sigma, learning_rate=tasa_aprendizaje,
                neighborhood_function=vecindad, topology=topology,
                 activation_distance=distance, random_seed=seed)
    
    #Weigh init
    if(pesos_init == 'pca'):
        som.pca_weights_init(data)
    elif(pesos_init == 'random'):   
        som.random_weights_init(data)

    print('Training som...')
    som.train(data, iteracciones, random_order=False, verbose=True)  
    session_data.set_modelo(som)                                                   

    print('Training Complete!')
    end = time.time()
    print('\t Elapsed Time:',str(end - start),'seconds')


    return 'Training Complete'








