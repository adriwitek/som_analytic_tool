# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from flask.globals import session
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
import views.plot_utils as pu
import math
import dash_table




#############################################################
#	                       LAYOUT	                        #
#############################################################

def train_som_view():

    '''
        A rule of thumb to set the size of the grid for a dimensionalityreduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.
    '''
    n_samples = session_data.get_train_data_n_samples()
    grid_recommended_size = ceil(  sqrt(5*sqrt(n_samples))  )

    # Formulario SOM    
    formulario_som = dbc.ListGroupItem([

                html.Div(
                    children=[ 
                        dbc.Button("Single Train", id="single_train_sel", className="mr-2", color="primary", outline=False ),
                        dbc.Button("Multiple Trains: Manual Selection", id="manual_train_sel", className="mr-2", color="primary", outline=True  ),
                        dbc.Button("Multiple Trains: Grid Search", id="grids_train_sel", className="mr-2", color="primary", outline=True  )
                    ],
                    style=pu.get_css_style_inline_flex()
                ),
                html.Br(),

                html.H4('Parameter Selection',className="card-title"  ),
                html.Div(id = 'som_form_div',
                        style=pu.get_css_style_inline_flex(),children=[

                        
                        dbc.Collapse(id = 'table_multiple_trains_collapse',
                                    is_open = False,
                                    children = get_multiple_train_table()
                        ),

                        html.Div(
                            style={'display': 'inline-block', 'text-align': 'left'},
                            children=[


                                html.H5(children='Vertical Grid Size:'),
                                dcc.Input(id="tam_eje_vertical", type="number", value=grid_recommended_size,step=1,min=1),

                                html.H5(children='Horizontal Grid Size'),
                                dcc.Input(id="tam_eje_horizontal", type="number", value=grid_recommended_size,step=1,min=1),

                                html.H5(children='Learning Rate'),
                                dcc.Input(id="tasa_aprendizaje_som", type="number", value="0.5",step=0.0001,min=0,max=5),


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
                                dcc.Input(id="iteracciones", type="number", value=session_data.get_train_data_n_samples(),
                                            step=1,min=1),

                                html.H5(children='Weights Initialization'),
                                dcc.Dropdown(
                                    id='dropdown_inicializacion_pesos',
                                    options=[
                                        {'label': 'PCA: Principal Component Analysis ', 'value': 'pca'},
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
                                            switch=True,
                                            id="check_semilla_som")]
                                ),
                                html.Div( id= 'div_semilla_som',
                                            children = [dcc.Input(id="seed_som", type="number", value="0",step=1,min=0, max=(2**32 - 1))],
                                            style= pu.get_css_style_hidden_visibility()
                                ),   

                                html.Hr(),
                                html.H5(children='Show Map qe Error evolution while training'),
                                html.Div( 
                                        [dbc.Checklist(
                                            options=[{"label": "Plot Evolution", "value": 1}],
                                            value=[],
                                            switch=True,
                                            id="check_qe_evolution_som")
                                        ]
                                ),

                                dbc.Collapse(   id = 'collapse_qe_evolution_som',
                                                is_open= False,
                                                children = [
                                                    dbc.Label(children='Plot evolution every  '),
                                                    dcc.Input(  id="input_qe_evolution_som", type="number",
                                                                value= math.ceil(0.1 * session_data.get_train_data_n_samples()),
                                                                #value= 100,
                                                                step=1,min=1,
                                                                max = session_data.get_train_data_n_samples()-1),
                                                    dbc.Label(children='   iterations'),
                                                ],
                                                style =pu.get_css_style_inline_flex_no_display()
                                ),
                                html.Hr(),
                        ]),
                ]),


                

                dbc.Collapse(id = 'single_train_collapse',
                    is_open = True,
                    children = [
                        dbc.Button("Train", id="train_button_som",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary")
                        #,dbc.Spinner(id='spinner_training',color="primary",fullscreen=False)],
                    ],
                    style={'textAlign': 'center'}
                ),
                html.H6(id='som_entrenado'),

                dbc.Collapse(id = 'manual_train_collapse',
                    is_open = False,
                    children = [
                        dbc.Button("Add", id="add_train_som",disabled= True, className="mr-2", color="primary"),
                        dbc.Button("Train", id="train_models_button_som",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary"),
                    ],
                    style={'textAlign': 'center'}
                ),
                html.H6(id='som_entrenado_2'),
                
                dbc.Collapse(id = 'grids_train_collapse',
                    is_open = False,
                    children = [
                        dbc.Button("Train with Grid Search", id="train_gs_button_som",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary"),
                    ],
                    style={'textAlign': 'center'}
                ),
                html.H6(id='som_entrenado_3'),


                
    ])




    layout = html.Div(children=[

        elements.navigation_bar,
        formulario_som,
    ])


    return layout








##################################################################
#                       AUX LAYOUT FUNCTIONS
##################################################################

def get_multiple_train_table():
            
    columns = []
    columns.append({'id': 'Horizontal Size'         , 'name': 'Hor. Size' })
    columns.append({'id': 'Vertical Size'           , 'name': 'Ver. Size' })
    columns.append({'id': "Learning Rate"           , 'name': "Learning Rate" })
    columns.append({'id': "Neighborhood Function"   , 'name': "Neighborhood Function" })
    columns.append({'id': "Distance Function"       , 'name': "Distance Function"})
    columns.append({'id': "Gaussian Sigma"          , 'name': "Gaussian Sigma"})
    columns.append({'id': "Max Iterations"          , 'name': "Max Iterations"})
    columns.append({'id': "Weights Initialization"  , 'name': "Weights Initialization"})
    columns.append({'id': "Topology"                , 'name': "Topology"})
    columns.append({'id': "Seed"                    , 'name': "Seed"})
    

    table =  dash_table.DataTable(	id = 'table_multiple_trains',
                                    columns = columns,
                                    row_deletable=True,
                                    editable=False,
                                    style_cell={      'textAlign': 'center',
                                                      'textOverflow': 'ellipsis',
                                                      'overflowX': 'auto'
                                    },
                                    #style_as_list_view=True,
                                    style_header={
                                            'backgroundColor': 'rgb(255, 255, 53)',
                                            'fontWeight': 'bold'
                                    },

                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        }
                                    ],
       
    )

    return table



def create_new_train_row(h_size, v_size, lr, nf, df,gs,mi,wi,t,s):
    row = {}
    row['Horizontal Size'] = h_size
    row['Vertical Size'] = v_size
    row['Learning Rate'] = lr
    row['Neighborhood Function'] = nf
    row['Distance Function'] = df
    row['Gaussian Sigma'] = gs
    row['Max Iterations'] = mi
    row['Weights Initialization'] = wi
    row['Topology'] = t
    row['Seed'] = s

    return row







#############################################################
#	                     CALLBACKS	                        #
#############################################################


#Select train or load model
@app.callback(  
                Output('single_train_sel','outline'),
                Output('manual_train_sel','outline'),
                Output('grids_train_sel','outline'),
                Output('single_train_collapse','is_open'),
                Output('manual_train_collapse','is_open'),
                Output('grids_train_collapse','is_open'),
                Output('table_multiple_trains_collapse', 'is_open'),
                Input('single_train_sel','n_clicks'),
                Input('manual_train_sel','n_clicks'),
                Input('grids_train_sel','n_clicks'),
                prevent_initial_call=True
)
def select_train_mode_som(n1,n2,n3):

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(button_id == 'single_train_sel'):
        session_data.set_som_multiple_trains( False)
        return False,  True, True,   True, False , False,    False
    elif(button_id == 'manual_train_sel' ):
        session_data.set_som_multiple_trains( True)
        return True,  False, True,   False, True , False,   True
    else: #grids_train_sel
        session_data.set_som_multiple_trains( False)
        return True,  True, False,   False, False , True,    False

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


# Checklist select plot qe evolution
@app.callback(
    Output('collapse_qe_evolution_som','is_open'),
    Input("check_qe_evolution_som", "value"),
    prevent_initial_call=True
    )
def plot_qe_evolution(check):

    if(check):
        return True
    else:
        return False



#Enable train som button
@app.callback(  Output('train_button_som','disabled'),
                Output('add_train_som','disabled'),
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
                Input("check_qe_evolution_som", "value"),
                Input("input_qe_evolution_som", "value")
)
def enable_train_som_button(tam_eje_vertical,tam_eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,
                            sigma,iteracciones,dropdown_inicializacion_pesos,  seed, check_semilla,
                            check_qe_evolution_som, input_qe_evolution_som):

    params  = [tam_eje_vertical,tam_eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,
                                    sigma,iteracciones,dropdown_inicializacion_pesos]

    if(check_semilla):
        params.append(seed)

    if( all(i is not None for i in params) and 
            (not check_qe_evolution_som or (check_qe_evolution_som and input_qe_evolution_som is not None) )    ):
        
        return False, False
    else:
        return True, True




@app.callback(  Output('dropdown_vecindad', 'options'),
                Output('dropdown_vecindad', 'value'),
                Input('dropdown_topology', 'value'),
                State('dropdown_vecindad', 'options'),
                State('dropdown_vecindad', 'value'),
                prevent_initial_call=True
)
def disable_triangular_with_hextopology(topology,options, valor_vecindad):

    if(topology == 'hexagonal'):
        options[3]['disabled']= True
        if(valor_vecindad == 'triangle'):
            valor_vecindad = 'gaussian'
    else:
        options[3]['disabled']= False
    
    return options,valor_vecindad



@app.callback(  Output('table_multiple_trains','data'),
                Input('add_train_som','n_clicks'),
                State('table_multiple_trains','rows'),
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
                State('table_multiple_trains','data'),
                prevent_initial_call=True
)
#append a new train to table
def create_new_train_som(n_clicks,rows, eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,
                        pesos_init, semilla, check_semilla, table_data):         

    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)
    iteracciones = int(iteracciones)
    if(check_semilla):
        seed = int(semilla)
        row = create_new_train_row(eje_horizontal, eje_vertical, tasa_aprendizaje, vecindad, distance,sigma,
                    iteracciones,pesos_init,topology,seed)
    else:
        row = create_new_train_row(eje_horizontal, eje_vertical, tasa_aprendizaje, vecindad, distance,sigma,
                    iteracciones,pesos_init,topology,'No')
        
    if(table_data is None):
        table_data = []
    table_data.append(row)
    return table_data


@app.callback(  Output('train_models_button_som', 'disabled'),
                Input('table_multiple_trains','data'),
                prevent_initial_call=True
)
#enable button
def enable_train_models_button_som(table_data):

    if(table_data is None or (not any(table_data)) ):
        return True
    else:
        return False


@app.callback(  Output('som_entrenado_2', 'children'),
                Input('train_models_button_som', 'n_clicks'),
                State('table_multiple_trains','data'),
                State("check_qe_evolution_som", "value"),
                State("input_qe_evolution_som", "value"),
                prevent_initial_call=True
)
def train_models_som(n_cliks, table_data, check_qe_evolution_som, input_qe_evolution_som):

    if(table_data is None or (not any(table_data)) ):
        return PreventUpdate

    session_data.set_n_of_models(len(table_data)) 

    for i, row in enumerate(table_data):
    
        session_data.set_current_training_model(i)
        eje_horizontal = row['Horizontal Size'] 
        eje_vertical = row['Vertical Size']  
        tasa_aprendizaje =   row['Learning Rate'] 
        vecindad =   row['Neighborhood Function'] 
        distance =   row['Distance Function'] 
        sigma =   row['Gaussian Sigma'] 
        iteracciones =   row['Max Iterations'] 
        pesos_init =   row['Weights Initialization'] 
        topology  =   row['Topology'] 
        seed  =   row['Seed'] 

        if(seed == 'No'):
            check_semilla = 0
            seed = None
        else:
            check_semilla = 1


        data = session_data.get_train_data()

        start = time.time()
        session_data.start_timer()

        # TRAINING


        

        som = minisom.MiniSom(x=eje_vertical, y=eje_horizontal, input_len=data.shape[1], sigma=sigma, learning_rate=tasa_aprendizaje,
                    neighborhood_function=vecindad, topology=topology,
                     activation_distance=distance, random_seed=seed)

        #Weigh init
        if(pesos_init == 'pca'):
            som.pca_weights_init(data)
        elif(pesos_init == 'random'):   
            som.random_weights_init(data)


        #print('Training som...')
        if(any(check_qe_evolution_som)):
            #print('dentrooooo')
            session_data.set_show_error_evolution(True)
            session_data.reset_error_evolution()
            map_qe = som.train(data, iteracciones, random_order=False, verbose=True,plot_qe_at_n_it = input_qe_evolution_som)  
        else:
            session_data.set_show_error_evolution(False)
            map_qe = som.train(data, iteracciones, random_order=False, verbose=True)  

        print('Training Complete!')
        end = time.time()
        #ojo en numpy: array[ejevertical][ejehorizontal] ,al contratio que en plotly
        session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,distance,sigma,iteracciones, 
                                            pesos_init,topology,check_semilla, seed, end - start, som = som, qe = map_qe)
        print('\t Elapsed Time:',str(end - start),'seconds')

        




    return 'Training Complete'









@app.callback(  Output('som_entrenado', 'children'),
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
                State("check_qe_evolution_som", "value"),
                State("input_qe_evolution_som", "value"),
                prevent_initial_call=True
)
def train_som(n_clicks,eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,
                pesos_init, semilla, check_semilla, check_qe_evolution_som, input_qe_evolution_som):


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
    

    som = minisom.MiniSom(x=eje_vertical, y=eje_horizontal, input_len=data.shape[1], sigma=sigma, learning_rate=tasa_aprendizaje,
                neighborhood_function=vecindad, topology=topology,
                 activation_distance=distance, random_seed=seed)
    
    #Weigh init
    if(pesos_init == 'pca'):
        som.pca_weights_init(data)
    elif(pesos_init == 'random'):   
        som.random_weights_init(data)


    #print('Training som...')
    if(any(check_qe_evolution_som)):
        #print('dentrooooo')
        session_data.set_show_error_evolution(True)
        session_data.reset_error_evolution()
        som.train(data, iteracciones, random_order=False, verbose=True,plot_qe_at_n_it = input_qe_evolution_som)  
    else:
        session_data.set_show_error_evolution(False)
        som.train(data, iteracciones, random_order=False, verbose=True)  
    session_data.set_modelos(som)                                                   

    print('Training Complete!')
    end = time.time()
    #ojo en numpy: array[ejevertical][ejehorizontal] ,al contratio que en plotly
    #TODO OJO VER SI ESTO AQUI PODRIA CAUSAR PROBLEMAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,distance,sigma,
                                            iteracciones, pesos_init,topology,check,seed, training_time = end)
    print('\t Elapsed Time:',str(end - start),'seconds')


    return 'Training Complete'








