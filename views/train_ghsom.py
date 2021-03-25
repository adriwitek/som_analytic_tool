
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
from  config.config import *


# Formulario GHSOM
formulario_ghsom =  dbc.ListGroupItem([
                    html.H4('Elección de parámetros',className="card-title"  ),

                    html.H5(children='Tau 1:'),
                    dcc.Slider(id='tau1_slider', min=0,max=1,step=0.00001,value=0.1),
                    dcc.Input(id="tau1", type="number", value="0.1",step=0.00001,min=0,max=1),

                    html.H5(children='Tau 2:'),
                    dcc.Input(id="tau2", type="number", value="0.0001",step=0.00001,min=0,max=1),
                    dcc.Slider(id='tau2_slider', min=0,max=1,step=0.00001,value=0.0001),

                    html.H5(children='Tasa de aprendizaje:'),
                    dcc.Input(id="tasa_aprendizaje", type="number", value="0.15",step=0.01,min=0,max=5),

                    html.H5(children='Decadencia:'),
                    dcc.Input(id="decadencia", type="number", value="0.95",step=0.01,min=0,max=1),   

                    html.H5(children='Sigma gaussiana:'),
                    dcc.Input(id="sigma", type="number", value="1.5",step=0.01,min=0,max=10),

                    html.H5(children='Número máximo de iteracciones:'),
                    dcc.Input(id="max_iter_ghsom", type="number", value="10",step=1),

                    html.H5(children='Épocas:'),
                    dcc.Input(id="epocas_ghsom", type="number", value="15",step=1,min=1),



                    html.Hr(),
                    html.Div( 
                        [dbc.Button("Entrenar", id="train_button_ghsom",href='analyze-ghsom-data',disabled= True, className="mr-2", color="primary")],
                        style={'textAlign': 'center'}
                    ),
                    #for training callback
                    html.Div(id='trained_ghsom_div',children=''),
                    


                ])






###############################   TABS     ##############################
tab1_content =  dbc.Card(color = 'light',children=[
        dbc.CardBody(
            dbc.ListGroup([
                # Dataset info
                dbc.ListGroupItem([
                    html.H4('Información del dataset:',className="card-title"  ),
                    elements.table
                ]),
                elements.model_selector,
                #Param election
                html.Div(id='modelo',children = formulario_ghsom )
            ],flush=True,),
        )
    ])


tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Tab 1"),
        dbc.Tab(tab2_content, label="Tab 2"),
        dbc.Tab(
            "This tab's content is never seen", label="Tab 3", disabled=True
        ),
    ]
)



###############################   LAYOUT     ##############################
layout = html.Div(children=[

    elements.navigation_bar,
    tabs,
])






                    #GHSOM
########################################################################################################


# Sync slider tau1
@app.callback(
    Output("tau1", "value"),
    Output("tau1_slider", "value"),
    Input("tau1", "value"),
    Input("tau1_slider", "value"), prevent_initial_call=True)
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
    Input("tau2_slider", "value"), prevent_initial_call=True)
def sync_slider_tau2(tau2, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = tau2 if trigger_id == "tau2" else slider_value
    return value, value


#Habilitar boton train ghsom
@app.callback(Output('train_button_ghsom','disabled'),
              Input('tau1','value'),
              Input('tau2','value'),
              Input('tasa_aprendizaje','value'),
              Input('decadencia','value'),
              Input('sigma','value'),
              Input('max_iter_ghsom','value'),
              Input('epocas_ghsom','value')
              )
def enable_train_ghsom_button(tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana,max_iter_ghsom, epocas_ghsom):
    '''Habilita el boton de train del ghsom

    '''
    if all(i is not None for i in [tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana, max_iter_ghsom, epocas_ghsom ]):
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
              prevent_initial_call=True )
def train_ghsom(n_clicks,tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana,epocas_ghsom, max_iter_ghsom):

    print('button clickedddddddddddd\n', flush=True)
    tau1 = float(tau1)
    tau2 = float(tau2)
    tasa_aprendizaje=float(tasa_aprendizaje)
    decadencia = float(decadencia)
    sigma_gaussiana = float(sigma_gaussiana)
    epocas_ghsom = int(epocas_ghsom)
    max_iter_ghsom = int(max_iter_ghsom)


    dataset = session_data.get_data()


    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]

    print('debug point 1')


    
    
    ghsom = GHSOM(input_dataset = data, t1= tau1, t2= tau2, learning_rate= tasa_aprendizaje, decay= decadencia, gaussian_sigma=sigma_gaussiana, growing_metric="qe")
    zero_unit = ghsom.train(epochs_number=epocas_ghsom, dataset_percentage=1, min_dataset_size=1, seed=0, grow_maxiter=max_iter_ghsom)
    session_data.set_ghsom_model_info_dict(tau1,tau2,tasa_aprendizaje,decadencia,sigma_gaussiana,epocas_ghsom,max_iter_ghsom)

    session_data.set_modelo(zero_unit)
    print('ENTRENAMIENTO DEL GHSOM FINALIZADO\n')
    print('zerounit:',zero_unit)

    
    return 'entrenamiento_completado'




