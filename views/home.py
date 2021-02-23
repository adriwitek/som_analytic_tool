# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import  views.elements as elements

#VENTANA PRINCIPAL 


def Home(): 
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
                        html.Div(id='output-data-upload_1',style={'textAlign': 'center'} ),
                        html.Div(id='output-data-upload_2',style={'textAlign': 'center'} ),
                        html.Div(id='n_samples',style={'textAlign': 'center'} ),
                        html.Div(id='n_features',style={'textAlign': 'center'} ),
                        html.Div( 
                            [dbc.Button("Analizar Datos", id="continue-button",disabled= True,href='/train-som', className="mr-2", color="primary",)],
                            style={'textAlign': 'center'}
                        )
                    ]),
                    #URL
                    dbc.ListGroupItem([
                        html.H4('URL',style={'textAlign': 'center'} ),
                        dbc.Input(type= 'url', placeholder="Link del dataset", bs_size="md", className="mb-3"),
                        html.Div( 
                            [dbc.Button("Analizar Datos", id="continue-button-url",disabled= True,href='/train-som', className="mr-2", color="primary",)],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ],flush=True,),


            )
        ]),

    
    
        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Cargar modelo pre-entrenado')),

            dbc.CardBody(
            

            
            )
        ])


    ])


    return layout





