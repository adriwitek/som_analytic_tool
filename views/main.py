# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px

from datetime import datetime


from views.app import app

#VENTANA PRINCIPAL 



#################Layout###############################
layout = html.Div(children=[

    html.Div(id="hidden_div_for_redirect_callback"),

    html.H1(children='Herramienta de análisis de datos con Mapas Auto-organizados'),
    html.Hr(),
    

    dcc.Upload( id='upload-data',
        children=html.Div([
            'Arrastrar y soltar o  ',
            html.A('Seleccionar archivo')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    html.Div(id='output-data-upload_1',style={'textAlign': 'center'} ),
    html.Div(id='output-data-upload_2',style={'textAlign': 'center'} ),
    html.Hr(),

    html.Button('Analizar Datos',id='continue-button',disabled= True, formTarget='/training_selection')

])
#################################################################



'''
def guarda_dataframe(contents):
    pass
    #almacenar aqui la info de contentes, qeu seria el dataset
'''



@app.callback(Output('hidden_div_for_redirect_callback', 'children'),
              Input('continue-button', 'n_clicks'), prevent_initial_call=True )
def redirect_to_training_selection(n_clicks):
     return dcc.Location(pathname="/training_selection", id="redirect")



@app.callback(Output('output-data-upload_1', 'children'),
              Output('output-data-upload_2', 'children'),
              Output('continue-button','disabled'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(contents, filename, last_modified):
    if contents is not None:
        output_1= 'Archivo: ' + filename
        output_2= 'Última Modificación: ' +  datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S')
        #guarda_dataframe(contents)

        return output_1, output_2,False
    else:
        return '','',True


