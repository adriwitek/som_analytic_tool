import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from views.app import app


import  views.elements as elements

from models.ghsom.GSOM import GSOM 
from models.ghsom.neuron.neuron_builder import NeuronBuilder
import json
import numpy as np

from  views.session_data import session_data
from  config.config import *

import time


# Formulario SOM
formulario_gsom =  dbc.ListGroupItem([
                    html.H4('Elección de parámetros',className="card-title"  ),

                    html.H5(children='Tamaño inicial del mapa(Eje vertical):'),
                    dcc.Input(id="tam_eje_vertical_gsom", type="number", value=5,step=1,min=1),

                    html.H5(children='Tamaño inicial del mapa(Eje horizontal):'),
                    dcc.Input(id="tam_eje_horizontal_gsom", type="number", value=5,step=1,min=1),

                    html.H5(children='Tau 1:'),
                    dcc.Input(id="tau1_gsom", type="number",step=0.0000001,min=0,max=1, value='0.0001'),
                    dcc.Slider(id='tau1_slider_gsom', min=0,max=1,step=0.0001,value=0.0001),

                    html.H5(children='Tasa de aprendizaje:'),
                    dcc.Input(id="tasa_aprendizaje_gsom", type="number", value="0.15",step=0.01,min=0,max=5),

                    html.H5(children='Decadencia:'),
                    dcc.Input(id="decadencia_gsom", type="number", value="0.95",step=0.01,min=0,max=1),   

                    html.H5(children='Sigma gaussiana:'),
                    dcc.Input(id="sigma_gsom", type="number", value="1.5",step=0.01,min=0,max=10),

                    html.H5(children='Número máximo de iteracciones:'),
                    dcc.Input(id="max_iter_gsom", type="number", value="10",step=1),

                    html.H5(children='Épocas:'),
                    dcc.Input(id="epocas_gsom", type="number", value="15",step=1,min=1),

                    html.H5(children='Semilla:'),
                    html.Div( 
                            [dbc.Checklist(
                                options=[{"label": "Seleccionar semilla", "value": 1}],
                                value=[],
                                id="check_semilla")]
                    ),
                    html.Div( id= 'div_semilla',
                                children = [dcc.Input(id="seed_gsom", type="number", value="0",step=1,min=0, max=(2**32 - 1))],
                                style={ "visibility": "hidden",'display':'none'}
                    ),    


                    html.Hr(),
                    html.Div( 
                        [dbc.Button("Entrenar", id="train_button_gsom",href='analyze-gsom-data' ,disabled= True, className="mr-2", color="primary")],
                        style={'textAlign': 'center'}
                    ),
                    
                    #for training callback
                    html.Div(id='testt_divv',children=''),


                ])




layout = html.Div(children=[

    elements.navigation_bar,
    elements.model_selector,
    formulario_gsom,
])




##################################################################
#                       CALLBACKS
##################################################################



# Checklist seleccionar semilla
@app.callback(
    Output('div_semilla','style'),
    Input("check_semilla", "value"),
    prevent_initial_call=True
    )
def select_seed(check):

    if(check):
        return {  'display': 'block'}
    else:
        return { "visibility": "hidden",'display':'none'}



# Sync slider tau1
@app.callback(
    Output("tau1_gsom", "value"),
    Output("tau1_slider_gsom", "value"),
    Input("tau1_gsom", "value"),
    Input("tau1_slider_gsom", "value"))
def sync_slider_tau1(tau1, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = tau1 if trigger_id == "tau1_gsom" else slider_value
    return value, value



#Habilitar boton train gsom
@app.callback(Output('train_button_gsom','disabled'),
              Input('tam_eje_vertical_gsom', 'value'),
              Input('tam_eje_horizontal_gsom', 'value'),
              Input('tau1_gsom','value'),
              Input('tasa_aprendizaje_gsom','value'),
              Input('decadencia_gsom','value'),
              Input('sigma_gsom','value'),
              Input('epocas_gsom','value'),
              Input('max_iter_gsom','value'),
              Input('seed_gsom','value'),
              Input("check_semilla", "value"),
            )
def enable_train_gsom_button(tam_eje_vertical_gsom,tam_eje_horizontal_gsom, tau1,tasa_aprendizaje_gsom,decadencia_gsom,sigma_gsom,epocas_gsom,max_iter_gsom,seed, check_semilla):
    '''Habilita el boton de train del gsom

    '''
    parametros = [tam_eje_vertical_gsom,tam_eje_horizontal_gsom,tau1,tasa_aprendizaje_gsom,decadencia_gsom,sigma_gsom,epocas_gsom,max_iter_gsom]
    if(check_semilla):
        parametros.append(seed)

    if all(i is not None for i in parametros ):
        return False
    else:
        return True





#Boton train ghsom
@app.callback(Output('testt_divv', 'children'),
              Input('train_button_gsom', 'n_clicks'),
              State('tam_eje_vertical_gsom', 'value'),
              State('tam_eje_horizontal_gsom', 'value'),
              State('tau1_gsom','value'),
              State('tasa_aprendizaje_gsom','value'),
              State('decadencia_gsom','value'),
              State('sigma_gsom', 'value'),
              State('epocas_gsom', 'value'),
              State('max_iter_gsom','value'),
              State('seed_gsom','value'),
              State("check_semilla", "value"),
              prevent_initial_call=True )
def train_gsom(n_clicks, tam_eje_vertical_gsom,tam_eje_horizontal_gsom ,tau_1,tasa_aprendizaje_gsom,decadencia_gsom,sigma,epocas_gsom,max_iter_gsom, semilla, check_semilla):
    
    tam_eje_vertical_gsom =int(tam_eje_vertical_gsom)
    tam_eje_horizontal_gsom = int(tam_eje_horizontal_gsom)
    tau_1 = float(tau_1)
    tasa_aprendizaje_gsom=float(tasa_aprendizaje_gsom)
    decadencia_gsom = float(decadencia_gsom)
    sigma_gausiana = float(sigma)
    epocas_gsom = int(epocas_gsom)
    max_iter_gsom = int(max_iter_gsom)
    if(check_semilla):
        seed = int(semilla)
    else:
        seed = None


    data = session_data.get_data()

    start = time.time()

  
    initial_map_size = (tam_eje_vertical_gsom, tam_eje_horizontal_gsom)
    neuron_builder = NeuronBuilder(1, growing_metric="qe")    #tau2= 1valur not used in gsom
    zero_unit = neuron_builder.zero_neuron(data)



    #__calc_initial_random_weights(self, seed):
    random_generator = np.random.RandomState(seed)
    random_weights = np.zeros(shape=(tam_eje_vertical_gsom, tam_eje_horizontal_gsom,data.shape[1]))
    for position in np.ndindex(tam_eje_vertical_gsom, tam_eje_horizontal_gsom):
        random_data_item = data[random_generator.randint(len(data))]
        random_weights[position] = random_data_item

    parent_quantization_error  = zero_unit.compute_quantization_error()
    print('parent_quantization_error', parent_quantization_error)
    print('tau1', tau_1)

    print('condi:',parent_quantization_error* tau_1)

    zero_unit.child_map = GSOM( initial_map_size,
                                parent_quantization_error,
                                tau_1,
                                data.shape[1],#esto es el numero de atributos
                                random_weights ,
                                data,
                                neuron_builder)
    
    
    
    '''
    print('epochs_number:',str(epocas_gsom),
                    'self.__gaussian_sigma',str(sigma_gausiana),
                    'self.__learning_rate',str(tasa_aprendizaje_gsom),
                    'self.__decay', str(decadencia_gsom),
                    'seed',str(seed),
                    'grow_maxiter', str(max_iter_gsom))
    '''
    #Train
    zero_unit.child_map.train(epocas_gsom,
                            sigma_gausiana,
                            tasa_aprendizaje_gsom,
                            decadencia_gsom,
                            dataset_percentage=1,
                            min_dataset_size=1,
                            seed=seed,
                            maxiter=max_iter_gsom)
    gsom = zero_unit.child_map
    
   
    #matriz_de_pesos_neuronas = __gmap_to_matrix(gsom.weights_map)


    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()
    session_data.set_gsom_model_info_dict(tam_eje_vertical,tam_eje_horizontal,tau_1,tasa_aprendizaje_gsom,decadencia_gsom,sigma_gausiana,epocas_gsom,max_iter_gsom, check_semilla, seed)
    session_data.set_modelo(zero_unit)


    end = time.time()
    print('ENTRENAMIENTO DEL GSOM FINALIZADO\n')
    print('Tiempo transcurrido en el entrenamiento:',str(end - start))

    #TODO
    return 'Gsom entrenado'



