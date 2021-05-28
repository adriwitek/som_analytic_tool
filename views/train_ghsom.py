# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements

from models.ghsom.GHSOM import GHSOM
from models.ghsom.GSOM import GSOM 

from  views.session_data import session_data
from config.config import MIN_TAU_STEP
from  config.config import *
import time


# Formulario GHSOM
formulario_ghsom =  dbc.ListGroupItem([
                    html.H4('Parameter Selection',className="card-title"  ),

                    html.Div(style={'textAlign': 'center'},children=[
                        html.Div(
                            style={'display': 'inline-block', 'text-align': 'left'},
                            children=[

                                html.H5(children='Tau 1'),
                                dcc.Input(id="tau1", type="number", value="0.1",step=MIN_TAU_STEP,min=0,max=1),
                                dcc.Slider(id='tau1_slider', min=0,max=1,step=0.0001,value=0.1),

                                html.H5(children='Tau 2'),
                                dcc.Input(id="tau2", type="number", value="0.0001",step=MIN_TAU_STEP,min=0,max=1),
                                dcc.Slider(id='tau2_slider', min=0,max=1,step=0.0001,value=0.0001),

                                html.H5(children='Learning Rate'),
                                dcc.Input(id="tasa_aprendizaje", type="number", value="0.15",step=0.01,min=0,max=5),

                                html.H5(children='Decadency'),
                                dcc.Input(id="decadencia", type="number", value="0.95",step=0.01,min=0,max=1),   

                                html.H5(children='Gaussian Sigma'),
                                dcc.Input(id="sigma", type="number", value="1.5",step=0.01,min=0,max=10),

                                html.H5(children='Max. Iterations'),
                                dcc.Input(id="max_iter_ghsom", type="number", value="10",step=1),

                                html.H5(children='Epochs per Iteration'),
                                dcc.Input(id="epocas_ghsom", type="number", value="15",step=1,min=1),

                                html.H5(children='Dissimilarity Function'),
                                dcc.Dropdown(
                                            id='dropdown_fun_desigualdad_ghsom',
                                            options=[
                                                {'label': 'Quantization Error', 'value': 'qe'},
                                                {'label': ' Average Quantization Error', 'value': 'mqe'}
                                            ],
                                            value='qe',
                                            searchable=False
                                ),

                                html.H5(children='Seed:'),
                                html.Div( 
                                        [dbc.Checklist(
                                            options=[{"label": "Select Seed", "value": 1}],
                                            value=[],
                                            id="check_semilla_ghsom")]
                                ),
                                html.Div( id= 'div_semilla_ghsom',
                                            children = [dcc.Input(id="seed_ghsom", type="number", value="0",step=1,min=0, max=(2**32 - 1))],
                                            style={ "visibility": "hidden",'display':'none'}
                                ),    


                                html.Hr(),
                                html.Div( 
                                    [dbc.Button("Train", id="train_button_ghsom",href=URLS['TRAINING_MODEL'],disabled= True, className="mr-2", color="primary")],
                                    style={'textAlign': 'center'}
                                ),

                                #for training callback
                                html.Div(id='trained_ghsom_div',children='')

                        ])
                    ])
                    
                ])





###############################   LAYOUT     ##############################
layout = html.Div(children=[

    elements.navigation_bar,
    formulario_ghsom,
])







##################################################################
#                       CALLBACKS
##################################################################


# Sync slider tau1
@app.callback(
    Output("tau1", "value"),
    Output("tau1_slider", "value"),
    Input("tau1", "value"),
    Input("tau1_slider", "value"))
def sync_slider_tau1(tau1, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = tau1 if trigger_id == "tau1" else slider_value
    return value, value


# Sync slider tau2
@app.callback(
    Output("tau2", "value"),
    Output("tau2_slider", "value"),
    Input("tau2", "value"),
    Input("tau2_slider", "value"))
def sync_slider_tau2(tau2, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = tau2 if trigger_id == "tau2" else slider_value
    return value, value



# Checklist seleccionar semilla
@app.callback(
    Output('div_semilla_ghsom','style'),
    Input("check_semilla_ghsom", "value"),
    prevent_initial_call=True
    )
def select_seed(check):

    if(check):
        return {  'display': 'block'}
    else:
        return { "visibility": "hidden",'display':'none'}


#Habilitar boton train ghsom
@app.callback(Output('train_button_ghsom','disabled'),
              Input('tau1','value'),
              Input('tau2','value'),
              Input('tasa_aprendizaje','value'),
              Input('decadencia','value'),
              Input('sigma','value'),
              Input('max_iter_ghsom','value'),
              Input('epocas_ghsom','value'),
              Input('dropdown_fun_desigualdad_ghsom','value'),
              Input('seed_ghsom','value'),
              Input("check_semilla_ghsom", "value"),
              )
def enable_train_ghsom_button(tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana,max_iter_ghsom, epocas_ghsom,
                                fun_desigualdad, seed, check_semilla):
    '''Habilita el boton de train del ghsom

    '''
    parametros = [tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana, max_iter_ghsom, epocas_ghsom, fun_desigualdad ]

    if(check_semilla):
        parametros.append(seed)

    if all(i is not None for i in parametros):
        return False
    else:
        return True



#Boton train ghsom
@app.callback(Output('trained_ghsom_div', 'value'),
              Input('train_button_ghsom', 'n_clicks'),
              State('tau1','value'),
              State('tau2','value'),
              State('tasa_aprendizaje','value'),
              State('decadencia','value'),
              State('sigma','value'),
              State('epocas_ghsom','value'),
              State('max_iter_ghsom','value'),
              State('dropdown_fun_desigualdad_ghsom','value'),
              State('seed_ghsom','value'),
              State("check_semilla_ghsom", "value"),
              prevent_initial_call=True )
def train_ghsom(n_clicks,tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana,epocas_ghsom, max_iter_ghsom,
                fun_desigualdad, semilla, check_semilla):

    tau1 = float(tau1)
    tau2 = float(tau2)
    tasa_aprendizaje=float(tasa_aprendizaje)
    decadencia = float(decadencia)
    sigma_gaussiana = float(sigma_gaussiana)
    epocas_ghsom = int(epocas_ghsom)
    max_iter_ghsom = int(max_iter_ghsom)
    
    if(check_semilla):
        seed = int(semilla)
        check = 1
    else:
        seed = None
        check = 0

    session_data.set_ghsom_model_info_dict(tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana,
                                            epocas_ghsom,max_iter_ghsom,fun_desigualdad,check,seed)


    #session_data.estandarizar_data()
    data = session_data.get_train_data() 

    start = time.time()
    session_data.start_timer()



    ghsom = GHSOM(input_dataset = data, t1= tau1, t2= tau2, learning_rate= tasa_aprendizaje, decay= decadencia,
                     gaussian_sigma=sigma_gaussiana, growing_metric=fun_desigualdad)

    print('Training ghsom...')

    zero_unit = ghsom.train(epochs_number=epocas_ghsom, dataset_percentage=1, min_dataset_size=1, seed=seed,
                             grow_maxiter=max_iter_ghsom)
    session_data.set_modelo(zero_unit)
    end = time.time()
    
    #print('zerounit:',zero_unit)

    print('Training Complete!')
    print('\t Elapsed Time:',str(end - start),'seconds')
    return 'Training Complete'



