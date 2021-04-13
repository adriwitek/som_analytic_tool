# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements
from  views.session_data import session_data

from  config.config import *


from os import listdir,makedirs
from os.path import isfile, join
import pickle







#############################################################
#	                       LAYOUT	                        #
#############################################################


def Training_selection(): 
    layout = html.Div(children=[

        html.Div(id="hidden_div_for_redirect_callback"),

        elements.navigation_bar,

        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Entrenar nuevo modelo')),

            dbc.CardBody(
                dbc.ListGroup([
                    
                    #SOM
                    dbc.ListGroupItem([
                        html.H4('SOM',style={'textAlign': 'center'} ),
                        html.Div( 
                            [dbc.Button("SOM", id="train_mode_som_button",href='/train-som', className="mr-2", color="primary",)],
                            style={'textAlign': 'center'}
                        )
                    ]),


                    #GSOM
                    dbc.ListGroupItem([
                        html.H4('GSOM',style={'textAlign': 'center'} ),
                        html.Div( 
                            [dbc.Button("GSOM", id="train_mode_gsom_button",href='/train-gsom', className="mr-2", color="primary",)],
                            style={'textAlign': 'center'}
                        )
                    ]),


                    #GHSOM
                    dbc.ListGroupItem([
                        html.H4('GHSOM',style={'textAlign': 'center'} ),
                        html.Div( 
                            [dbc.Button("GHSOM", id="train_mode_ghsom_button",href='/train-ghsom', className="mr-2", color="primary",)],
                            style={'textAlign': 'center'}
                        )
                    ]),


                ],flush=True,),


            )
        ]),

    
    
        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Cargar modelo pre-entrenado')),

            dbc.CardBody(
                dbc.ListGroup([

                    # Modelos guardados en la app
                    dbc.ListGroupItem([
                        html.H4('Modelos guardados',className="card-title" , style={'textAlign': 'center'} ),
                        html.Div(style={'textAlign': 'center'},
                                children=[
                                    dcc.Dropdown(
                                        id='modelos_guardados_en_la_app_dropdown',
                                        options=get_app_saved_models(),
                                        value='',
                                        searchable=False
                                        #,style={'width': '35%'}
                        
                                    ),

                                    dbc.Button("Cargar modelo", id="load_saved_model_button",disabled= True,href='/', className="mr-2", color="primary"),
                                    html.Div(id='hidden_div_for_load_model',style={'textAlign': 'center'} ),


                                ]
                            
                            
                        )
                       




                
                    ]),
                ],flush=True,),


            )
        ]),


    ])


    return layout










#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################


 
def get_app_saved_models():

    makedirs(DIR_SAVED_MODELS, exist_ok=True)

    onlyfiles = [f for f in listdir(DIR_SAVED_MODELS) if isfile(join(DIR_SAVED_MODELS, f))]

    options = []  # must be a list of dicts per option
    for f in onlyfiles:
        if f.endswith('.pickle'):
            options.append({'label' : f, 'value': f})
    return options







#############################################################
#	                     CALLBACKS	                        #
#############################################################




#Habilitar boton load_saved_model
@app.callback(Output('load_saved_model_button','disabled'),
              Input('modelos_guardados_en_la_app_dropdown','value'),
              prevent_initial_call=True
              )

def enable_load_saved_model_button(values):
    if ( values ):
        return False
    else:
        return True


#load selected model
@app.callback(Output('hidden_div_for_redirect_callback', 'children'),
              Input('load_saved_model_button', 'n_clicks'),
              State('modelos_guardados_en_la_app_dropdown','value'),
              prevent_initial_call=True )
def load_selected_model(n_clicks,filename):


    # Load data (deserialize)
    with open(DIR_SAVED_MODELS + filename, 'rb') as handle:
        unserialized_data = pickle.load(handle)
        model_type= unserialized_data[0]
        model_info = unserialized_data[1]
        session_data.set_modelo(unserialized_data[2])

    
    if  model_type ==  'som':
        session_data.set_som_model_info_dict_direct(model_info)
        return dcc.Location(pathname=URLS['ANALYZE_SOM_URL'], id="redirect")
    elif model_type ==   'gsom':
        session_data.set_gsom_model_info_dict_direct(model_info)
        return dcc.Location(pathname=URLS['ANALYZE_GSOM_URL'], id="redirect")
    elif model_type ==   'ghsom':
        session_data.set_ghsom_model_info_dict_direct(model_info)
        return dcc.Location(pathname=URLS['ANALYZE_GHSOM_URL'], id="redirect")
    else:   #it something goes worng 
        return dcc.Location(pathname="/", id="redirect")





'''
ESTO HACERLO PARA CUANDO EL DATASET ES NONE REDIRIGIR A HOME!!!!!!!!!!!!!!!!!!

@app.callback(Output('hidden_div_for_redirect_callback', 'children'),
              Input('continue-button', 'n_clicks'),
              prevent_initial_call=True )
def update_app_content_view(n1,n2,n3,n4):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (trigger_id == "continue-button"):
        return dcc.Location(pathname="/train-som", id="redirect")
    elif (trigger_id == "seleccion_modelo_som"):
            return dcc.Location(pathname="/train-som", id="redirect")
    elif(trigger_id == "seleccion_modelo_gsom"):
        return dcc.Location(pathname="/train-gsom", id="redirect")
    elif(trigger_id == "seleccion_modelo_gsom"):
        return dcc.Location(pathname="/train-ghsom", id="redirect")
    else:
        return dcc.Location(pathname="/", id="redirect")
'''