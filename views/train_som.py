# -*- coding: utf-8 -*-

import dash
from dash_bootstrap_components._components.Collapse import Collapse
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

import models.sklearn_interface_for_som.sklearn_interface_for_som as sksom
from models.sklearn_interface_for_som.scoring_functions import  score_mqe, score_topographic_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


#############################################################
#	                       LAYOUT	                        #
#############################################################



# Form containing input hyperparams
def form_train_som(grid_recommended_size):
    children = [

                    
                     #This 4 params may be oculted if param tuning is seleted. If then, they will be a range, not a single input
                    dbc.Collapse(id = 'collapse_size',
                                is_open = True,
                                children =[
                                        html.H5(children='Vertical Grid Size'),
                                        dcc.Input(id="tam_eje_vertical", type="number", value=grid_recommended_size,step=1,min=1),
                                        html.H5(children='Horizontal Grid Size'),
                                        dcc.Input(id="tam_eje_horizontal", type="number", value=grid_recommended_size,step=1,min=1),
                                ]
                    ),


                    dbc.Collapse(id = 'collapse_distancef',
                                is_open = True,
                                children =[
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
                                ]
                    ),

                    dbc.Collapse(id = 'collapse_weightsini',
                                is_open = True,
                                children =[
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
                                ]
                    ),
                

                    

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

                    html.H5(children='Gaussian Sigma'),
                    dcc.Input(id="sigma", type="number", value="1.5",step=0.000001,min=0,max=10),
                                
                    html.H5(children='Max Iterations', id="tooltip_target"),
                    dbc.Tooltip(
                        "1 Iteration per Data Sample,  "
                        "If it is bigger, samples will be repeated in rounds.",
                        target = 'iteracciones',
                        placement = 'right'
                    ),
                    dcc.Input(id="iteracciones", type="number", value=session_data.get_train_data_n_samples(),
                                step=1,min=1),

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

                    
                    dbc.Collapse(id = 'collapse_show_qe_evolution',
                                    is_open=True,
                                    children = [
                                            html.Hr(),
                                            html.H5(children='Show Map QE Error evolution while training'),
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
                                    ]
                    ),
                   
                    dbc.Modal(
                            [
                                dbc.ModalHeader("Done"),
                                dbc.ModalBody("This model will be avaible at Analyze Page, after training other models in Multi-Train Tab!"),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="close_modal_psearch_to_multitrain_tab", className="ml-auto")
                                ),
                            ],
                            id="modal_psearch_to_multitrain_tab",
                            centered=True,
                            is_open= False,
                    ),

                    
                   
            ]

    return children




def train_som_view():

    '''
        A rule of thumb to set the size of the grid for a dimensionalityreduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.
    '''
    n_samples = session_data.get_train_data_n_samples()
    grid_recommended_size = ceil(  sqrt(5*sqrt(n_samples))  )

    grid_start_default_size = ceil(0.8 * grid_recommended_size)
    grid_stop_default_size = grid_recommended_size + ceil(0.2 * grid_recommended_size)
    grid_step_default_value = ceil(grid_stop_default_size/grid_start_default_size)
    if((grid_step_default_value % 2) == 0):
        grid_stop_default_size = grid_start_default_size * (grid_step_default_value+1)
    else:
        grid_stop_default_size = grid_start_default_size * grid_step_default_value

    # Formulario SOM    
    formulario_som = dbc.ListGroupItem([

                html.Div(
                    children=[ 
                        dbc.Button("Single Train", id="single_train_sel", className="mr-2", color="primary", outline=False ),
                        dbc.Button("Multi-Train", id="manual_train_sel", className="mr-2", color="primary", outline=True  ),
                        dbc.Button("Grid Hyperparameters Search", id="grids_train_sel", className="mr-2", color="primary", outline=True  ),
                        dbc.Button("Random Hyperparameters Search", id="random_train_sel", className="mr-2", color="primary", outline=True  )

                    ],
                    style=pu.get_css_style_inline_flex()
                ),
                html.Br(),

                html.H4('Hyperparameter Selection',className="card-title"  ),
                html.Div(id = 'som_form_div',
                        style=pu.get_css_style_inline_flex_align_2_elements(),children=[

                        dbc.Collapse(id = 'table_multiple_trains_collapse',
                                    is_open = False,
                                    children = get_multiple_train_table()
                        ),

                        #Only showed for param search tuning
                        dbc.Collapse(id = 'collapse_param_tuning',
                                is_open = False,
                                children =[

                                        html.Hr(),
                                        html.H4( 'Params to make a search' ),
                                        html.Br(),
                                        #Vertical and horizontal Size Range
                                        html.Div(
                                            [   dbc.Button("Add",id='button_add_size_psearch', outline=True, color="success", className="mr-1"),
                                                html.H5(id = 'add_tag_size_gs_or_rands',children='Square Grid Size'),
                                            ],
                                            style = pu.get_css_style_inline_flex_align_flex_start_no_wrap(),
                                        ),
                                        dbc.Collapse(   id ='collapse_range_size_gsearch',
                                                        is_open = False,
                                                        style= pu.get_css_style_center() ,
                                                        children = [
                                                            html.Br(),
                                                            dbc.Label('Start:'),
                                                            dbc.Input(id = 'input_start_size' ,type='number', min = 1, step = 1, value = grid_start_default_size),                              
                                                            dbc.Label('Stop:'),
                                                            dbc.Input(id = 'input_stop_size' ,type='number', min = 2, step = 1,value =grid_stop_default_size ),                              
                                                            dbc.Label('Step:'),
                                                            dbc.Input(id = 'input_step_size' , type='number',min = 1, step = 1, value = grid_step_default_value),                              
                                                        ]
                                        ),

                                        dbc.Collapse(   id ='collapse_range_size_randsearch',
                                                        is_open = False,
                                                        style= pu.get_css_style_center() ,
                                                        children = [
                                                            html.Br(),
                                                            html.Div(
                                                                [   dbc.Label('Vertical Size Start:'),
                                                                    dbc.Input(id = 'input_vstart_size_randsearch' ,type='number', min = 1, step = 1, value = grid_start_default_size),                              
                                                                    dbc.Label('Vertical Size Stop:'),
                                                                    dbc.Input(id = 'input_vstop_size_randsearch' ,type='number', min = 2, step = 1,value =grid_stop_default_size ),  
                                                                ],
                                                                style = pu.get_css_style_inline_flex_no_display()
                                                            ), 

                                                            html.Div(
                                                                [   dbc.Label('Horizontal Size Start:'),
                                                                    dbc.Input(id = 'input_hstart_size_randsearch' ,type='number', min = 1, step = 1, value = grid_start_default_size),                              
                                                                    dbc.Label('Horizontal Size Stop:'),
                                                                    dbc.Input(id = 'input_hstop_size_randsearch' ,type='number', min = 2, step = 1,value =grid_stop_default_size ),  
                                                                ],
                                                                style = pu.get_css_style_inline_flex_no_display()
                                                            ), 
                                                                                       
                                                            dbc.Label('Number of hyperparameter settings to sample'),
                                                            dbc.Input(id = 'input_n_iter_randsearch' , type='number',min = 1, step = 1, value = 10),                              
                                                        ]
                                        ),
                                        html.Br(),


                                       
                                        # Distance Funcion Range
                                        html.Div(
                                            [   dbc.Button("Add",id='button_add_distancef_psearch', outline=True, color="success", className="mr-1"),
                                                html.H5(children='Distance Function'),
                                            ],
                                            style = pu.get_css_style_inline_flex_align_flex_start_no_wrap(),
                                        ),
                                        dbc.Collapse(   id ='collapse_range_distancef_psearch',
                                                        is_open = False,
                                                        style= pu.get_css_style_inline_flex_no_display() ,
                                                        children = [
                                                            html.Br(),
                                                            dbc.Label('Values to search with:'),
                                                            dcc.Dropdown(
                                                                id='dropdown_distance_psearch',
                                                                multi=True,
                                                                options=[
                                                                    {'label': 'Euclidean', 'value': 'euclidean'},
                                                                    {'label': 'Cosine', 'value': 'cosine'},
                                                                    {'label': 'Manhattan', 'value': 'manhattan'},
                                                                    {'label': 'Chebyshev', 'value': 'chebyshev'}
                                                                ],
                                                                value=['euclidean','cosine', 'manhattan', 'chebyshev'],
                                                                searchable=False
                                                                #style={'width': '35%'}
                                                            ),
                                                                                       
                                                        ]
                                        ),
                                        html.Br(),

                                        #Weights Init Range
                                        html.Div(
                                            [   dbc.Button("Add",id='button_add_weightsini_psearch', outline=True, color="success", className="mr-1"),
                                                html.H5(children='Weights Initialization'),
                                            ],
                                            style = pu.get_css_style_inline_flex_align_flex_start_no_wrap(),
                                        ),
                                        dbc.Collapse(   id ='collapse_range_weightsini_psearch',
                                                        is_open = False,
                                                        style= pu.get_css_style_inline_flex_no_display() ,
                                                        children = [
                                                            html.Br(),                          
                                                            dbc.Label('Values to search with:'),
                                                            dcc.Dropdown(
                                                                id='dropdown_weightsini_psearch',
                                                                multi=True,
                                                                options=[
                                                                    {'label': 'PCA: Principal Component Analysis ', 'value': 'pca'},
                                                                    {'label': 'Random', 'value': 'random'},
                                                                    {'label': 'No Weight Initialization ', 'value': 'no_init'}
                                                                ],
                                                                value=['pca','random', 'no_init'],
                                                                searchable=False
                                                                #style={'width': '35%'}
                                                            ),
                                                                                       
                                                        ]
                                        ),

                                        html.Hr(),

                                        html.H5(children='Scoring metric to select best model'),
                                        dcc.Dropdown(
                                                                id='dropdown_scoring_metric_psearch',
                                                                multi=False,
                                                                options=[
                                                                    {'label': 'Mean Quantization Error', 'value': 'mqe'},
                                                                    {'label': 'Topographic Error', 'value': 'tp'}
                                                                ],
                                                                value='mqe',
                                                                searchable=False
                                                                #style={'width': '35%'}
                                                            ),


                                ]
                        ),


                        html.Div(
                            id = 'form_train_som',
                            style={'display': 'inline-block', 'text-align': 'left'},
                            children=form_train_som(grid_recommended_size)
                        ),
                ]),


                

                dbc.Collapse(id = 'single_train_collapse',
                    is_open = True,
                    children = [
                        dbc.Button("Train", id="train_button_som",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary")
                        #,dbc.Spinner(id='spinner_training',color="primary",fullscreen=False)],
                    ],
                    style=pu.get_css_style_center()
                ),
                html.H6(id='som_entrenado'),

                dbc.Collapse(id = 'manual_train_collapse',
                    is_open = False,
                    children = [
                        dbc.Button("Add", id="add_train_som",disabled= True, className="mr-2", color="success"),
                        dbc.Button("Train", id="train_models_button_som",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary"),
                    ],
                    style=pu.get_css_style_center()
                ),
                html.H6(id='som_entrenado_2'),
                
                #Grid Search
                dbc.Collapse(id = 'collapse_gridsearch_train_form',
                    is_open = False,
                    children = [
                        dcc.Loading(id='loading',
                                    type='dot',
                                    children=[
                                        dbc.Button("Grid Search for Optimum Hyperparameters", id="gridsearch_params_button_som",
                                        disabled= True, className="mr-2", color="warning"),
                                    ]
                        ),
                        
                    ],
                    style=pu.get_css_style_center()
                ),



                #Random Search
                dbc.Collapse(id = 'collapse_randsearch_train_form',
                    is_open = False,
                    children = [
                        dcc.Loading(id='loading',
                                    type='dot',
                                    children=[
                                        dbc.Button("Rand Search for Optimum Hyperparameters", id="randsearch_params_button_som",
                                        disabled= True, className="mr-2", color="warning"),
                                    ]
                        ),
                    ],
                    style=pu.get_css_style_center()
                ),


                #Table with best found hyperparams
                dbc.Collapse(   id= 'collapse_optimum_found_params',
                                is_open= False,
                                style=pu.get_css_style_center(),
                                children=[
                                    html.Br(),
                                    html.Br(),
                                    html.Div(id = 'div_table_search_found_params', children =pu.create_simple_table([],[], 'table_search_found_params') ),
                                    html.Br(),
                                    html.Div([
                                                dbc.Button("Add Params to MultiTrain Tab", id="psearch_addtomulti_train_button",disabled= True, className="mr-2", color="success"),
                                                dbc.Button("Discard HyperParameters", id="discard_search_params_button",disabled= True, className="mr-2", color="danger"),
                                                dbc.Button("Analyze Model with current Hyperparameters", id="psearch_analyze_button",href = URLS['ANALYZE_SOM_URL'],
                                                            disabled= True, className="mr-2", color="primary"),  
                                            ],
                                            style = pu.get_css_style_inline_flex_no_display()
                                    ),
                                ]
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



#For param tuning search
def add_data_table_search_found_params( h_size = None, v_size= None,df= None, wi= None, mqe= None, tp= None, data = None):

    columns = []
    if(data is None):
        data = []
    row = {}

    if(mqe is not None):
        columns.append({'id': 'mqe'         , 'name': 'Average Mean Quantization Error' })
        row['mqe'] = mqe
    if(tp is not None):
        columns.append({'id': 'tp'         , 'name': ' Average Topographic Error' })
        row['tp'] = tp
    if(h_size is not None):
        columns.append({'id': 'Horizontal Size'         , 'name': 'Hor. Size' })
        row['Horizontal Size'] = h_size
    if(v_size is not None):
        columns.append({'id': 'Vertical Size'           , 'name': 'Ver. Size' })
        row['Vertical Size'] = v_size
    if(df is not None):
        columns.append({'id': "Distance Function"       , 'name': "Distance Function"})
        row['Distance Function'] = df
    if(wi is not None):
        columns.append({'id': "Weights Initialization"  , 'name': "Weights Initialization"})
        row['Weights Initialization'] = wi

    data.append(row)

    return data, columns
   





#############################################################
#	                     CALLBACKS	                        #
#############################################################





#start,stop,step values coordintaed in param tuning tab
@app.callback( 
                Output('input_start_size', 'invalid'),
                Output('input_stop_size', 'invalid'),
                Output('input_step_size', 'invalid'),
                Input('input_start_size', 'value'),
                Input('input_stop_size', 'value'),
                Input('input_step_size', 'value'),
                prevent_initial_call=True
)
def sync_start_stop_step_inputs(start,stop,step):

    
    if(start is None or stop is None or step is None):
        return True, True, True
    elif(start >=stop):
        return  True, True, True
    elif(stop<(start + step)):
        return  True, True, True
    elif( (stop - start)%step != 0 ):
        return  True, True, True
    else:
        return False, False, False


#validate_input_vstart_size_randsearch
@app.callback(  Output('input_vstart_size_randsearch', 'invalid'),
                Output('input_vstop_size_randsearch', 'invalid'),
                Input('input_vstart_size_randsearch', 'value'),
                Input('input_vstop_size_randsearch', 'value'),
                prevent_initial_call=True
)
def validate_input_vstart_size_randsearch(start,stop):
    if(start is None or stop is None ):
        return True,True
    elif(start >=stop):
        return  True,True
    else:
        return False, False


#validate_input_hstart_size_randsearch
@app.callback(  Output('input_hstart_size_randsearch', 'invalid'),
                Output('input_hstop_size_randsearch', 'invalid'),
                Input('input_hstart_size_randsearch', 'value'),
                Input('input_hstop_size_randsearch', 'value'),
                prevent_initial_call=True
)
def validate_input_hstart_size_randsearch(start,stop):
    if(start is None or stop is None ):
        return True,True
    elif(start >=stop):
        return  True,True
    else:
        return False, False


#validate_input_hstart_size_randsearch
@app.callback(  Output('input_n_iter_randsearch', 'invalid'),
                Input('input_n_iter_randsearch', 'value'),
                prevent_initial_call=True
)
def validate_input_hstart_size_randsearch(value):
    if(value is None or value <1 ):
        return True
    else:
        return False

        
#enable randsearch_params_button_som button
@app.callback(  Output('randsearch_params_button_som', 'disabled'),
                Input('input_vstart_size_randsearch', 'invalid'),
                Input('input_vstop_size_randsearch', 'invalid'),
                Input('input_hstart_size_randsearch', 'invalid'),
                Input('input_hstop_size_randsearch', 'invalid'),
                Input('input_n_iter_randsearch', 'invalid'),
                Input('gridsearch_params_button_som', 'disabled'),
                prevent_initial_call=True
)
def enable_randsearch_button(i1,i2,i3,i4,i5, disabled_button):

    if( i1 or i2 or i3 or i4 or i5 or disabled_button  ):
        return True
    else:
        False





#Toggles size param search menu in param search option
@app.callback(  Output('collapse_range_size_gsearch','is_open'),
                Output('collapse_range_size_randsearch','is_open'),

                Output('collapse_size','is_open'),
                Output('button_add_size_psearch', 'outline'),
                Output('button_add_size_psearch', 'children'),
                Output('add_tag_size_gs_or_rands', 'children'),

            
                Input('button_add_size_psearch', 'n_clicks'),
                Input('grids_train_sel', 'outline'),
                Input('random_train_sel' , 'outline'),
                State('button_add_size_psearch', 'outline'),
                prevent_initial_call=True
)
def toggle_size_psearch(n_clicks,outline_tab,outline_tab_2, button_outline ):
    if(outline_tab and outline_tab_2):
        return False,False, True, True, 'Add', 'Square Grid Size'
    ctx = dash.callback_context
    triggered_list =     trigger_id = ctx.triggered[0]["prop_id"].split(".")
    trigger_id = triggered_list[0]
    prop_id = triggered_list[1]
    #trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(not outline_tab):#Grid Search
        if(button_outline and trigger_id == 'button_add_size_psearch' and prop_id == 'n_clicks'):#Not added --> Change to  Added
            return True,False, False, False, 'Added', 'Square Grid Size'
        else:
            return False,False, True, True, 'Add', 'Square Grid Size'
    else:#Random Search
        if(button_outline and trigger_id == 'button_add_size_psearch'  and prop_id == 'n_clicks'):#Not added --> Change to  Added
            return False,True, False, False, 'Added', 'Grid Size Sampling Ranges'
        else:
            return False,False, True, True, 'Add', 'Grid Size Sampling Ranges'



#Toggles distance fun param search menu in param search option
@app.callback(  Output('collapse_range_distancef_psearch','is_open'),
                Output('collapse_distancef','is_open'),
                Output('button_add_distancef_psearch', 'outline'),
                Output('button_add_distancef_psearch', 'children'),
                Input('button_add_distancef_psearch', 'n_clicks'),
                Input('grids_train_sel', 'outline'),
                Input('random_train_sel', 'outline'),
                State('button_add_distancef_psearch', 'outline'),
                prevent_initial_call=True
)
def toggle_distancef_psearch(n_clicks,outline_tab,outline_tab_2, button_outline ):
    if(outline_tab and  outline_tab_2):
        return False, True, True, 'Add'
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if(trigger_id == 'button_add_distancef_psearch'):
        if(button_outline):#Not added --> Change to  Added
            return True, False, False, 'Added'
        else:
            return False, True, True, 'Add'
    else:
        raise PreventUpdate


#Toggles weights init param search menu in param search option
@app.callback(  Output('collapse_range_weightsini_psearch','is_open'),
                Output('collapse_weightsini','is_open'),
                Output('button_add_weightsini_psearch', 'outline'),
                Output('button_add_weightsini_psearch', 'children'),
                Input('button_add_weightsini_psearch', 'n_clicks'),
                Input('grids_train_sel', 'outline'),
                Input('random_train_sel', 'outline'),
                State('button_add_weightsini_psearch', 'outline'),
                prevent_initial_call=True
)
def toggle_weightsinit_psearch(n_clicks,outline_tab,outline_tab_2, button_outline ):
    if(outline_tab and outline_tab_2 ):
        return False, True, True, 'Add'
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if(trigger_id == 'button_add_weightsini_psearch'):
        if(button_outline):#Not added --> Change to  Added
            return True, False, False, 'Added'
        else:
            return False, True, True, 'Add'
    else:
        raise PreventUpdate



#Select train or load model and Param option is_open form with param ranges
@app.callback(  
                Output('single_train_sel','outline'),
                Output('manual_train_sel','outline'),
                Output('grids_train_sel','outline'),
                Output('random_train_sel','outline'),

                Output('single_train_collapse','is_open'),
                Output('manual_train_collapse','is_open'),
                Output('collapse_gridsearch_train_form','is_open'),
                Output('collapse_randsearch_train_form','is_open'),

                Output('table_multiple_trains_collapse', 'is_open'),
                Output('collapse_param_tuning', 'is_open'),
                Output('collapse_show_qe_evolution', 'is_open'),
                Input('single_train_sel','n_clicks'),
                Input('manual_train_sel','n_clicks'),
                Input('grids_train_sel','n_clicks'),
                Input('random_train_sel','n_clicks'),
                prevent_initial_call=True
)
def select_train_mode_som(n1,n2,n3,n4):

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(button_id == 'single_train_sel'):
        session_data.set_som_multiple_trains( False)
        return False,  True, True,True,   True, False , False,False,    False, False, True
    elif(button_id == 'manual_train_sel' ):
        session_data.set_som_multiple_trains( True)
        return True,  False, True,True,   False, True , False,False,   True,False,True
    else: #grids_train_sel or random_train_sel
        session_data.set_som_multiple_trains( False)
        if(button_id == 'grids_train_sel'):
            return True,  True, False,True,   False, False , True,False,    False, True,False
        else:
            return True,  True,True, False,   False, False , False, True,   False, True,False



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
#append a new train to table: this can be done on multiple manual train tab or after param search(with the bests params found)
def create_new_train_som(n_clicks,rows, eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,
                        pesos_init, semilla, check_semilla, table_data):         

    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)
    iteracciones = int(iteracciones)
    #ctx = dash.callback_context
    #trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    '''
    if(trigger_id == 'psearch_addtomulti_train_button'):
        if(param_table_data is None or len(param_table_data)==0):
            raise PreventUpdate
        else:
           distance = param_table_data[0]['Distance Function']
           pesos_init = param_table_data[0]['Weights Initialization']
           if('Horizontal Size' in param_table_data[0] ):
               eje_vertical = param_table_data[0]['Vertical Size']
               eje_horizontal= param_table_data[0]['Horizontal Size']
    ''' 
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

    #session_data.reset_som_model_info_dict()
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
            session_data.set_show_error_evolution(True)
            session_data.reset_error_evolution()
            map_qe = som.train(data, iteracciones, random_order=False, verbose=True,plot_qe_at_n_it = input_qe_evolution_som)  
        else:
            session_data.set_show_error_evolution(False)
            map_qe = som.train(data, iteracciones, random_order=False, verbose=True)  

        print('\t-->Training Complete!')
        end = time.time()
        #ojo en numpy: array[ejevertical][ejehorizontal] ,al contratio que en plotly
        session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,distance,sigma,iteracciones, 
                                            pesos_init,topology,check_semilla, seed, end - start, som = som, mqe = map_qe,
                                             training_time = (end - start))
        print('\t\tElapsed Time:',str(end - start),'seconds')



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
    session_data.reset_som_model_info_dict()
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

    print('\t-->Training som...')
    #Random order =False due to data alrady shuffled
    if(any(check_qe_evolution_som)):
        session_data.set_show_error_evolution(True)
        session_data.reset_error_evolution()
        som.train(data, iteracciones, random_order=False, verbose=True,plot_qe_at_n_it = input_qe_evolution_som)  
    else:
        session_data.set_show_error_evolution(False)
        som.train(data, iteracciones, random_order=False, verbose=True)  
    session_data.set_modelos(som)                                                   

    print('\t-->Training Complete!')
    end = time.time()
    #ojo en numpy: array[ejevertical][ejehorizontal] ,al contratio que en plotly
    session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,distance,sigma,
                                            iteracciones, pesos_init,topology,check,seed, training_time = (end - start))
    print('\t\tElapsed Time:',str(end - start),'seconds')
    return 'Training Complete'





#############PARAM SEARCH CALLBACKS##########


#enable gridsearch_params_button_som button
@app.callback(  Output('gridsearch_params_button_som', 'disabled'),
                Input('input_start_size', 'invalid'),
                Input('button_add_size_psearch', 'outline'),
                Input('button_add_distancef_psearch', 'outline'),
                Input('button_add_weightsini_psearch', 'outline'),
                Input('dropdown_distance_psearch','value'),
                Input('dropdown_weightsini_psearch', 'value'),
                Input('input_start_size', 'value'),
                Input('input_stop_size', 'value'),
                Input('input_step_size', 'value'),
                prevent_initial_call=True
)
def enable_gridsearch_params_button_som(invalid_start, outline1, outline2,outline3,dd1,dd2,start,stop,step):

    #we must have mora than just one value
    cond = 0
    if(not outline1):
        if(  invalid_start):
            return True
        elif(start + step == stop):
            cond += 1
        else:
            cond += 2

    if(not outline2):
        if(dd1 is None or len(dd1) == 0 ):
            return True
        elif(not outline2 and len(dd1)==1 ):
            cond += 1
        else:
            cond += 2

    if( not outline3 ):
        if(dd2 is None or len(dd2) == 0):
            return True
        elif(not outline3 and len(dd2)==1 ):
            cond += 1
        else:
            cond += 2
    
    if(cond >=2 ):
       return False
    else:
        return True


#Grid and Random Search Button
@app.callback(  Output('div_table_search_found_params', 'children'),
                Output('collapse_optimum_found_params', 'is_open'),
                Output('gridsearch_params_button_som', 'color'),# This output is only for showin loading animation while searching
                Output('randsearch_params_button_som', 'color'),# This output is only for showin loading animation while searching
                Output("modal_psearch_to_multitrain_tab", "is_open"),

                Input('gridsearch_params_button_som', 'n_clicks'),
                Input('randsearch_params_button_som', 'n_clicks'),
                Input('psearch_addtomulti_train_button','n_clicks'),
                Input('close_modal_psearch_to_multitrain_tab','n_clicks'),
                Input('discard_search_params_button','n_clicks'),

                Input('single_train_sel', 'n_clicks'),
                Input('manual_train_sel', 'n_clicks'),
                Input('grids_train_sel', 'n_clicks'),
                Input('random_train_sel', 'n_clicks'),

                State('button_add_size_psearch', 'outline'),
                State('button_add_distancef_psearch', 'outline'),
                State('button_add_weightsini_psearch', 'outline'),

                State('dropdown_distance_psearch','value'),
                State('dropdown_weightsini_psearch', 'value'),
                State('input_start_size', 'value'),
                State('input_stop_size', 'value'),
                State('input_step_size', 'value'),
                State('dropdown_scoring_metric_psearch','value'),

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

                State("input_vstart_size_randsearch", "value"),
                State("input_vstop_size_randsearch", "value"),
                State("input_hstart_size_randsearch", "value"),
                State("input_hstop_size_randsearch", "value"),
                State("input_n_iter_randsearch", "value"),

                State('grids_train_sel','outline'),
                State('random_train_sel','outline'),
                prevent_initial_call=True
)
def param_search(  n1,n2,n3,n4,n5, 
                n6,n7,n8,n9,
                outline1, outline2,outline3, dd1,dd2,start,stop,step,scoring_metric,
                eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad, topology, distance,sigma,iteracciones,
                pesos_init, semilla, check_semilla,
                v_start, v_stop,h_start, h_stop, n_iter_randsearch,
                outline_tab , outline_tab_2 ):

    #open_modal = False
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(trigger_id == 'single_train_sel' or trigger_id == 'manual_train_sel'): #Hide/show collapse found params
        return dash.no_update, False, dash.no_update,dash.no_update, dash.no_update
    elif(trigger_id == 'grids_train_sel' or trigger_id == 'random_train_sel'):#Hide/show collapse found params
        if(outline_tab and outline_tab_2 ):
            return dash.no_update, False, dash.no_update,dash.no_update, dash.no_update,
        else:
            raise PreventUpdate
    elif(trigger_id == 'close_modal_psearch_to_multitrain_tab'):
        return dash.no_update, dash.no_update, dash.no_update,dash.no_update, False
    elif(trigger_id == 'psearch_addtomulti_train_button'  ):
        return dash.no_update, dash.no_update, dash.no_update,dash.no_update,True
    elif(trigger_id == 'discard_search_params_button'):
        session_data.del_last_som_model_info_dict()
        return  pu.create_simple_table([], [], 'table_search_found_params'), False, dash.no_update ,dash.no_update,False
    
   
    parameters = {}
    table_h_size, table_v_size,table_df, table_wi,table_mqe, table_tp = None,None,None,None,None,None
    tasa_aprendizaje=float(tasa_aprendizaje)
    sigma = float(sigma)
    iteracciones = int(iteracciones)
    if(check_semilla):
        seed = int(semilla)
    else:
        seed = None

    if(not outline1 and trigger_id == 'gridsearch_params_button_som'):# Grid Search  wit squaresize
        parameters['square_grid_size'] =[ i for i in range(start, (stop+step), step) ]
    elif(not outline1 and trigger_id == 'randsearch_params_button_som'): #Rando search ver. and hor. range sizes
        parameters['ver_size'] = [i for i in range(v_start,v_stop+1 ,1 )]
        parameters['hor_size'] = [i for i in range(h_start,h_stop+1 ,1 )]
        #print('DEBUG RAND SERACH PARAMS ISEZ ADDED')


    if(not outline2):
        parameters['activation_distance'] = dd1
    else:
        parameters['activation_distance'] = [distance]

    if(not outline3):
        parameters['weights_init'] = dd2
    else:
        parameters['weights_init'] = [pesos_init]

    estimador = sksom.SOM_Sklearn(  ver_size =eje_vertical,hor_size = eje_horizontal, sigma=sigma, 
                                    learning_rate=tasa_aprendizaje,
                                    neighborhood_function=vecindad, topology=topology,
                                    num_iteration=iteracciones, random_seed=seed,
                                    activation_distance=distance,weights_init=pesos_init )

    
    scoring ={'tp': score_topographic_error,'mqe': score_mqe}
    score_fun = scoring[scoring_metric]
    data = session_data.get_train_data()
    if(trigger_id == 'gridsearch_params_button_som'):#Apply grid search
        gs = GridSearchCV(estimador, parameters, n_jobs = -1, scoring= score_fun, refit = True)
        print('\t -->Applying Hyperparameter Grid Search...')
    else: #Rand Search
        gs = RandomizedSearchCV(estimador, parameters,n_iter=n_iter_randsearch, n_jobs = -1, scoring= score_fun,
                                 refit = True)
        print('\t -->Applying Hyperparameter Random Search...')
    gs.fit(data)
    print('\t -->Hyperparameter Search Complete!')

    #Save best model since if later analyze this model button is used

    session_data.set_show_error_evolution(False)
    session_data.set_modelos(gs.best_estimator_.som_model) 

    s_params = gs.best_params_

    table_df=  s_params['activation_distance']
    table_wi = s_params['weights_init']
    if(scoring_metric == 'mqe'):
        table_mqe =  abs(gs.best_score_) 
    else:
        table_tp =  abs(gs.best_score_) 

    #Best params
    if('square_grid_size' in s_params):#grid search size
        table_h_size=s_params['square_grid_size']
        table_v_size=s_params['square_grid_size']
        session_data.set_som_model_info_dict(table_v_size,table_h_size,tasa_aprendizaje,vecindad,s_params['activation_distance'],sigma,
                                            iteracciones, s_params['weights_init'],topology,check_semilla,seed)
    elif('ver_size' in s_params):#random search size
        table_h_size=s_params['hor_size']
        table_v_size=s_params['ver_size']
        session_data.set_som_model_info_dict(table_v_size,table_h_size,tasa_aprendizaje,vecindad,s_params['activation_distance'],sigma,
                                            iteracciones, s_params['weights_init'],topology,check_semilla,seed)
    else:
        session_data.set_som_model_info_dict(eje_vertical,eje_horizontal,tasa_aprendizaje,vecindad,s_params['activation_distance'],sigma,
                                            iteracciones, s_params['weights_init'],topology,check_semilla,seed)
        

    data,columns = add_data_table_search_found_params(table_h_size, table_v_size,table_df, table_wi,table_mqe, table_tp)
    s_table = pu.create_simple_table(data, columns, 'table_search_found_params')
    #print('Results',gs.cv_results_)
    return s_table, True, "warning", "warning", False


#Enable analyze model after param search button
@app.callback(  Output('psearch_analyze_button', 'disabled'),
                Output('psearch_addtomulti_train_button', 'disabled'),
                Output('discard_search_params_button', 'disabled'),
                Input('table_search_found_params','data'),
                prevent_initial_call=True
)
def enable_analyze_model( data):
    if(data is None or len(data)==0):
        return True, True, True
    else:
        return False,False,False
