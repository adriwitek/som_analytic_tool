# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements


import io
from io import BytesIO
from datetime import datetime
import base64
from pandas import read_csv
import numpy as np

from  views.session_data import session_data
from  config.config import *







#############################################################
#	                       LAYOUT	                        #
#############################################################

def Home(): 

    session_data.clean_session_data()
    layout = html.Div(children=[

        html.Div(id="hidden_div_for_redirect_callback"),

        elements.navigation_bar,
        elements.cabecera,

        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Seleccionar dataset')),

            dbc.CardBody(
                dbc.ListGroup([
                    # Archivo Local
                    dbc.ListGroupItem([
                        html.H4('Archivo Local',className="card-title" , style={'textAlign': 'center'} ),
                        dcc.Upload( id='upload-data', children=html.Div(['Arrastrar y soltar o  ', html.A('Seleccionar archivo')]),
                                            style={'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px'},
                                            # Allow multiple files to be uploaded
                                            multiple=False),
                        #info showed when the dataset its loaded
                        html.Div(id='info_dataset_div',style={'textAlign': 'center', "visibility": "hidden",'display':'none'} ,children = div_info_dataset('','', '', '') ), 
                        html.Div(id='hidden_div',children= '' ), 

                        html.Div( 
                            [dbc.Button("Analizar Datos", id="continue-button",disabled= True,href=URLS['TRAINING_SELECTION_URL'], className="mr-2", color="primary",)],
                            style={'textAlign': 'center'}
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

 
def div_info_dataset(filename,fecha_modificacion, n_samples, n_features):
    return html.Div(id='output_uploaded_file',children=[
                html.P(children= 'Archivo:  ' + filename, style={'textAlign': 'center'} ),
                html.P(children= 'Última modificación: ' + fecha_modificacion, style={'textAlign': 'center'} ),
                html.P(children= 'Número de datos:  ' + n_samples , style={'textAlign': 'center'} ),
                html.P(children= 'Número de Atributos:  ' + n_features , style={'textAlign': 'center'} ),


                html.Div(id='output-data-upload_1',style={'textAlign': 'center'} ),
                html.Div(id='output-data-upload_2',style={'textAlign': 'center'} ),
                html.Div(id='n_samples',style={'textAlign': 'center'} ),
                html.Div(id='n_features',style={'textAlign': 'center'} ),
                dbc.RadioItems(
                    options=[
                        {"label": "Clases Discretas", "value": 1},
                        {"label": "Clases Continuas", "value": 2},
                    ],
                    value=1,
                    id="radio_discrete_continuous",
                ),
            ])
                      









#############################################################
#	                     CALLBACKS	                        #
#############################################################


#Discrete/cont. data
@app.callback(Output('hidden_div', 'children'),
              Input('radio_discrete_continuous', 'value'),
              prevent_initial_call=True)
def update_target_type(radio_option):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (trigger_id == "radio_discrete_continuous"):
        
        if(radio_option == 1):
            session_data.set_discrete_data(True)
        else:
            session_data.set_discrete_data(False)

        return  ''



 #Carga la info del dataset en el home
@app.callback(Output('info_dataset_div', 'children'),
              Output('info_dataset_div', 'style'),
              Output('continue-button','disabled'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'), 
                prevent_initial_call=True)
def update_output( contents, filename, last_modified):
    '''Carga el dataset en los elementos adecuados

    '''
    if contents is not None:
        

        #Esta carga tiene que ser asi
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                dataset = read_csv(io.StringIO(decoded.decode('utf-8')))
                

            #elif 'xls' in filename:
                # Assume that the user uploaded an excel file
            #    df = pd.read_excel(io.BytesIO(decoded))

        except Exception as e:
            print(e)
            return html.Div([ 'There was an error processing this file.'])
        

        data = dataset.to_numpy()
        n_samples, n_features=data.shape
        columns_names = list(dataset.head())
        session_data.set_dataset(data,columns_names)


        #N_FEATURES = N-1 because of the target column
        return (div_info_dataset(filename,
                                datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S'),
                                str(n_samples),
                                str(n_features - 1)), 
                {'textAlign': 'center',  'display': 'block'},
                False 
                )

    else: #TODO PREVENT init call makes this its never called
        return  div_info_dataset('','', '', '') ,{'textAlign': 'center', "visibility": "hidden"},True




