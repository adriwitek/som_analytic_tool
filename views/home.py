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

from pandas import read_csv, get_dummies, concat
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
                        dcc.Upload( id='upload-data', children=html.Div(['Arrastrar y soltar o  ', html.A('Seleccionar archivo  (.csv)')]),
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

                        html.Div(id='hidden_div_forcontinue',children = ''),
                        html.Div( 
                            [dbc.Button("Analizar Datos", id="continue_button_home",disabled= True,
                            href=URLS['TRAINING_SELECTION_URL'], className="mr-2", color="primary")],
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
                html.P(children= 'Número de Datos:  ' + n_samples , style={'textAlign': 'center'} ),
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

                html.Hr(),
                dbc.Checklist(  options=[{"label": "Aplicar One Hot Encoding", "value": 0}],
                                            value=[],
                                            id="check_onehot"),


                html.Div(id='div_onehot_menu',children=''),

                html.Hr(),


            ])
                      


def get_onehot_childrendiv_menu():

    children = [


        html.H5("Seleccionar atributos a los que aplicar One Hot:"),
        dcc.Dropdown(
            id='dropdown_atrib_names_home',
            options=session_data.get_dataset_atrib_names_dcc_dropdown_format(),
            multi=True
        ),

        dbc.Checklist(  options=[{"label": "Considerar también valores nulos", "value": 0}],
                                            value=[],
                                            id="check_nanvalues_onehot"
        ),
        

        dbc.Button("Aplicar One Hot", id="apply_onehot_button", className="mr-2", color="primary"),

        html.P(id='onehot_aplicado',children = '')


    ]

    return children





#############################################################
#	                     CALLBACKS	                        #
#############################################################


#Apply One Hot
@app.callback(Output('onehot_aplicado', 'children'),
              Output('apply_onehot_button', 'disabled'),
              Output('check_onehot','options'),
              Output('check_nanvalues_onehot','options'),
              Output('dropdown_atrib_names_home','disabled'),
              Input('apply_onehot_button', 'n_clicks'),
              State('dropdown_atrib_names_home','value'),
              State('check_onehot','options'),
              State('check_nanvalues_onehot','options'),
              State('check_nanvalues_onehot','value'),
              prevent_initial_call=True )
def apply_onehot(n_clicks, names,options, options_nan,check_nan = False ):

    if(not names):
        return 'Debes seleccionar al menos un atributo',False, options,options_nan,False

    df = session_data.pd_dataframe
    #print('antes\n', df)

    for n in names:
        # use pd.concat to join the new columns with your original dataframe
        df = concat( [df,get_dummies(df[n], prefix=str(n), dummy_na=check_nan) ],axis=1)
        # now drop the original 'country' column (you don't need it anymore)
        df.drop([n],axis=1, inplace=True)

    session_data.pd_dataframe = df
    #print('despues\n',session_data.pd_dataframe)
    options1=[{"label": "Considerar también valores nulos", "value": 0, "disabled": True}]
    options2=[{"label": "Considerar también valores nulos", "value": 0,"disabled": True}]
    options3=[{"disabled": True}]

    return 'One Hot Enconding Aplicado satisfactoriamente.',True, options1,options2,True


#Show One Hot Encoding Menu
@app.callback(Output('div_onehot_menu', 'children'),
              Input('check_onehot', 'value'),
              prevent_initial_call=True )
def show_onehot_menu(check_onehot):
    if(check_onehot):
        return get_onehot_childrendiv_menu()
    else:
        return ''


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
              Output('continue_button_home','disabled'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'), 
                prevent_initial_call=True)
def update_output( contents, filename, last_modified):
    '''Carga el dataset en los elementos adecuados

    '''
    if contents is not None:
        
        show_file_info_style =  {'textAlign': 'center',  'display': 'block'}

        #Esta carga tiene que ser asi
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                dataset = read_csv(io.StringIO(decoded.decode('utf-8')))
                session_data.pd_dataframe = dataset

            #TODO EXCEL FILES
            #elif 'xls' in filename:
                # Assume that the user uploaded an excel file
            #    df = pd.read_excel(io.BytesIO(decoded))

            else:
                return html.Div([ 'ERROR: Formato de archivo no admitido.']),show_file_info_style,True

        except Exception as e:
            #print(e)
            return html.Div([ 'Ha ocurrido un error procesando el archivo.']), show_file_info_style,True
        
        #TODO BORRAR
        #data = dataset.to_numpy()

        n_samples, n_features=dataset.shape
        if(n_samples == 0):
            return html.Div([ 'ERROR: El fichero no contiene ningún ejemplo']),show_file_info_style,True
        elif(n_features<=1):
            return html.Div([ 'ERROR: El fichero no contiene ningún atributo']),show_file_info_style,True


        columns_names = list(dataset.head())
        session_data.set_columns_names(columns_names)
        #TODO BORRAR
        #session_data.set_dataset(data,columns_names)


        #N_FEATURES = N-1 because of the target column
        return (div_info_dataset(filename,
                                datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S'),
                                str(n_samples),
                                str(n_features - 1)), 
                show_file_info_style,
                False 
                )

    else: 
        return  div_info_dataset('','', '', '') ,{'textAlign': 'center', "visibility": "hidden"},True




#Boton de continuar
@app.callback(Output('hidden_div_forcontinue', 'children'),
              Input('continue_button_home', 'n_clicks'),
              prevent_initial_call=True)
def analizar_datos_home( n_clicks ):

    pddataset = session_data.pd_dataframe
    data = pddataset.to_numpy()
    columns_names = list(pddataset.head())
    del session_data.pd_dataframe
    session_data.pd_dataframe = None
    session_data.set_dataset(data,columns_names)

    return ' '