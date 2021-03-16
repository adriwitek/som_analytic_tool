import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
import dash
import  views.elements as elements
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil
import numpy as np


from  views.session_data import session_data
from  config.config import *


fig = go.Figure()




def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),
        html.Hr(),
        html.Div(children=[ 

            #Card Mapa neurona winners
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_1"))
                ),
                dbc.Collapse(id="collapse_1",children=
                    dbc.CardBody(children=[ 
                        html.Div([dcc.Graph(id='winners_map',figure=fig)],
                          style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="ver", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),





            #Card: Frecuencias de activacion
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de frecencias de activacion",color="link",id="button_collapse_4"))
                ),
                dbc.Collapse(id="collapse_4",children=
                    dbc.CardBody(children=[
                        html.H5("Fecuencias de activacion:"),

                        html.Div([dcc.Graph(id='frequency_map',figure=fig)],
                          style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="frequency_map_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                        
                    ]),
                ),
            ]),




            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de componentes",color="link",id="button_collapse_2"))
                ),
                dbc.Collapse(id="collapse_2",children=
                    dbc.CardBody(children=[
                        html.H5("Seleccionar atributos para mostar:"),
                        dcc.Dropdown(
                            id='dropdown_atrib_names',
                            options=session_data.get_nombres_atributos(),
                            multi=True
                        ),
                        html.Div( 
                            [dbc.Checklist(
                                options=[{"label": "Seleccionar todos", "value": 1}],
                                value=[],
                                id="check_seleccionar_todos_mapas"),
                            dbc.Button("Ver Mapas de Componentes", id="ver_mapas_componentes_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),
                        html.Div(id='component_plans_figures_div', children=[''],
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                        )

                    ]),
                ),
            ]),



            #Card: U Matrix
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Matriz U",color="link",id="button_collapse_3"))
                ),
                dbc.Collapse(id="collapse_3",children=
                    dbc.CardBody(children=[
                        #METER AQUI LO QUE SEAA
                    html.H5("U-Matrix"),
                    html.H6("Returns the distance map of the weights.Each cell is the normalised sum of the distances betweena neuron and its neighbours. Note that this method usesthe euclidean distance"),
                    html.Div( 
                            [dbc.Button("Ver", id="umatrix_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                    ),

                    html.Div(id='umatrix_figure_div', children=[''],
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                    )

                    ])

                    ),
            ])



        ])

        
        
        
    
        
    



    ])




    ###############################   LAYOUT     ##############################
    layout = html.Div(children=[

        elements.navigation_bar,
        body,
    ])

    return layout







##################################################################
#                       CALLBACKS
##################################################################

@app.callback(
    [Output(f"collapse_{i}", "is_open") for i in range(1, 5)],
    [Input(f"button_collapse_{i}", "n_clicks") for i in range(1, 5)],
    [State(f"collapse_{i}", "is_open") for i in range(1, 5)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4, is_open1, is_open2,is_open3,is_open4):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4
    elif button_id == "button_collapse_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4
    elif button_id == "button_collapse_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4
    elif button_id == "button_collapse_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4
    return False, False, False,False


#Habilitar boton ver_mapas_componentes_button
@app.callback(Output('ver_mapas_componentes_button','disabled'),
              Input('dropdown_atrib_names','value')
            )
def enable_ver_mapas_componentes_button(values):
    if ( values ):
        return False
    else:
        return True





#Actualizar mapas de componentes
@app.callback(Output('component_plans_figures_div','children'),
              Input('ver_mapas_componentes_button','n_clicks'),
              State('dropdown_atrib_names','value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_fig(click,names):


    som = session_data.get_modelo()
    with open(SESSION_DATA_FILE_DIR) as json_file:
        datos_entrenamiento = json.load(json_file)

    tam_eje_x = datos_entrenamiento['som_tam_eje_x'] 
    tam_eje_y = datos_entrenamiento['som_tam_eje_y'] 
    nombres_columnas = datos_entrenamiento['columns_names']
    nombres_atributos = nombres_columnas[0:len(nombres_columnas)-1]
    lista_de_indices = []


    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    

    pesos = som.get_weights()

    traces = []

    for i in lista_de_indices:
        
        figure= go.Figure(layout= {"height": 300,'width' : 300, 'title': nombres_atributos[i] },
                          data=go.Heatmap(z=pesos[:,:,i].tolist(),showscale= True)                                                      
        ) 

        id ='graph-{}'.format(i)

        traces.append(
            html.Div(children= dcc.Graph(id=id,figure=figure)
            ) 
        )



    print('render finalizado')
    return traces
  










# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names','value'),
    Input("check_seleccionar_todos_mapas", "value"),
    prevent_initial_call=True
    )
def on_form_change(check):

    if(check):
        with open(SESSION_DATA_FILE_DIR) as json_file:
            datos_entrenamiento = json.load(json_file)

        nombres = datos_entrenamiento['columns_names']
        atribs= nombres[0:len(nombres)-1]
        return atribs
    else:
        return []




    
    ############################################
    #               MAPA DE FRECUENCIAS


    
#Actualizar mapas de frecuencias
@app.callback(Output('frequency_map','figure'),
              Input('frequency_map_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_mapa_frecuencias_fig(click):

    som = session_data.get_modelo() 
    som =  session_data.get_modelo()
    dataset = session_data.get_data()
    data = dataset[:,:-1]
 
    #frequencies is a np matrix
    frequencies = som.activation_response(data)
    figure= go.Figure(layout= {'title': 'Mapa de frecuencias absolutas'},
                          data=go.Heatmap(z=frequencies.tolist(),showscale= True)                              
    ) 
    return figure
   
  


      
#U-matrix
@app.callback(Output('umatrix_figure_div','children'),
              Input('umatrix_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_umatrix(click):

    som = session_data.get_modelo()
    umatrix = som.distance_map()
 
    figure= go.Figure(layout= {'title': 'Matriz U'},
                          data=go.Heatmap(z=umatrix.tolist(),showscale= True)                              
    ) 


    print('render finalizado')
    return  html.Div(children= dcc.Graph(id='graph_u_matrix',figure=figure))
