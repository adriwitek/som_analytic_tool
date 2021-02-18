# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import  views.elements as elements

#VENTANA PRINCIPAL 



layout = html.Div(children=[

    html.Div(id="hidden_div_for_redirect_callback"),
    elements.navigation_bar,
    elements.cabecera,
    '''
    second_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This card also has some text content and not much else, but "
                "it is twice as wide as the first card."
            ),
            dbc.Button("Go somewhere", color="primary"),
        ]
    ),
    '''

    html.H2('Seleccionar dataset'),

    html.Hr(),
    html.H4('Archivo Local', style={'textAlign': 'center'} ),
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
    html.Div( 
        [dbc.Button("Analizar Datos", id="continue-button",disabled= True,external_link='/training_selection', className="mr-2", color="primary",)],
        style={'textAlign': 'center'}
    ),
    

    html.Hr(),
    html.H4('URL',style={'textAlign': 'center'} ),
    dbc.Input(type= 'url', placeholder="Link del dataset", bs_size="md", className="mb-3"),
    html.Div( 
        [dbc.Button("Analizar Datos", id="continue-button-url",disabled= True,external_link='/training_selection', className="mr-2", color="primary",)],
        style={'textAlign': 'center'}
    ),
    html.Hr(),


    html.H2('Cargar modelo pre-entrenado'),




    html.Div(id='output-data-upload_1',style={'textAlign': 'center'} ),
    html.Div(id='output-data-upload_2',style={'textAlign': 'center'} ),
    html.Div(id='n_samples',style={'textAlign': 'center'} ),
    html.Div(id='n_features',style={'textAlign': 'center'} ),
    html.Hr()

    

])







