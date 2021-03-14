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
from collections import Counter



from  views.session_data import Sesion
from  config.config import *



fig = go.Figure()




def analyze_gsom_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),
        html.Hr(),
        html.Div(children=[ 

            #Card Mapa neurona winners
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_gsom_1"))
                ),
                dbc.Collapse(id="collapse_gsom_1",children=
                    dbc.CardBody(children=[ 
                        html.Div([dcc.Graph(id='winners_map_gsom',figure=fig)],
                          style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="ver_winners_map_gsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),



            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de componentes",color="link",id="button_collapse_gsom_2"))
                ),
                dbc.Collapse(id="collapse_gsom_2",children=
                    dbc.CardBody(children=[
                        dbc.CardBody(children=[
                            html.H5("Seleccionar atributos para mostar:"),
                            dcc.Dropdown(
                                id='dropdown_atrib_names_gsom',
                                options=Sesion.get_nombres_atributos(),
                                multi=True
                            ),
                            html.Div( 
                                [dbc.Checklist(
                                    options=[{"label": "Seleccionar todos", "value": 1}],
                                    value=[],
                                    id="check_seleccionar_todos_mapas_gsom"),
                                dbc.Button("Ver Mapas de Componentes", id="ver_mapas_componentes_button_gsom", className="mr-2", color="primary")],
                                style={'textAlign': 'center'}
                            ),
                            html.Div(id='component_plans_figures_gsom_div', children=[''],
                                    style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                            )

                        ]),
                    ]),
                ),
            ]),



            #Card: U Matrix
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("aaaaaaa",color="link",id="button_collapse_gsom_3"))
                ),
                dbc.Collapse(id="collapse_gsom_3",children=
                    dbc.CardBody(children=[

                    ])

                    ),
            ]),


             #Card: Frecuencias de activacion
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("aaaaaaa",color="link",id="button_collapse_gsom_4"))
                ),
                dbc.Collapse(id="collapse_gsom_4",children=
                    dbc.CardBody(children=[
                  
                        
                    ]),
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
    [Output(f"collapse_gsom_{i}", "is_open") for i in range(1, 5)],
    [Input(f"button_collapse_gsom_{i}", "n_clicks") for i in range(1, 5)],
    [State(f"collapse_gsom_{i}", "is_open") for i in range(1, 5)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4, is_open1, is_open2,is_open3,is_open4):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_gsom_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4
    elif button_id == "button_collapse_gsom_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4
    elif button_id == "button_collapse_gsom_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4
    elif button_id == "button_collapse_gsom_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4
    return False, False, False,False









#Actualizar mapas de frecuencias
@app.callback(Output('winners_map_gsom','figure'),
              Input('ver_winners_map_gsom_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_winner_map_gsom(click):


    #Carga de datos
    with open(SESSION_DATA_FILE_DIR) as json_file:
        session_data = json.load(json_file)

    tam_eje_x = session_data['som_tam_eje_x'] 
    tam_eje_y = session_data['som_tam_eje_y'] 
    dataset = Sesion.data
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]


    zero_unit = Sesion.modelo
    gsom = zero_unit.child_map
    

    #visualizacion

    data_to_plot = np.empty([tam_eje_x ,tam_eje_y],dtype=object)
    positions={}

    # Getting winnig neurons for each data element
    for i,d in enumerate(data):
        winner_neuron = gsom.winner_neuron(d)[0][0]
        r, c = winner_neuron.position
        if((r,c) in positions):
            positions[(r,c)].append(dataset[i][-1]) 
        else:
            positions[(r,c)] = []
            positions[(r,c)].append(dataset[i][-1]) 

    print('posiciones:', positions)

    # Obtener clases representativas de cada neurona
    for i in range(tam_eje_x):
        for j in range(tam_eje_y):
            if((i,j) in positions):

                if(session_data['discrete_data'] ):        #showing the class more represented in each neuron
                    c = Counter(positions[(i,j)])
                    data_to_plot[i][j] = c.most_common(1)[0][0]
                else: #continuos data: mean of the mapped values in each neuron
                    data_to_plot[i][j]  = np.mean(positions[(i,j)])
    
            else:
                data_to_plot[i][j] = None

   
        

    print('data_to_plot:',data_to_plot)
    
    #type= heatmap para mas precision
    #heatmapgl
    trace = dict(type='heatmap', z=data_to_plot, colorscale = 'Jet')
    data=[trace]

    # Here's the key part - Scattergl text! 
    


    data.append({'type': 'scattergl',
                    'mode': 'text',
                    #'x': x_ticks,
                    #'y': y_ticks,
                    'text': 'a'
                    })
    
    layout = {}
    #layout['xaxis'] = {'range': [-0.5, tam_eje_x]}
    layout['width'] = 700
    layout['height']= 700
    annotations = []
    fig = dict(data=data, layout=layout)

    #Poner o no etiquetas...

    print('\nVISUALIZACION:gsom renderfinalizado\n')

    return fig
   
  



# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names_gsom','value'),
    Input("check_seleccionar_todos_mapas_gsom", "value"),
    prevent_initial_call=True
    )
def on_form_change(check):

    if(check):
        with open(SESSION_DATA_FILE_DIR) as json_file:
            session_data = json.load(json_file)

        nombres = session_data['columns_names']
        atribs= nombres[0:len(nombres)-1]
        return atribs
    else:
        return []
