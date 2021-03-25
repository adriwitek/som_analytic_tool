import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
import dash
import  views.elements as elements
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
from plotly.subplots import make_subplots
from math import ceil
import numpy as np
from collections import Counter
from datetime import datetime


from  views.session_data import session_data
from  config.config import *
import pickle

from  os.path import normpath 
from re import search 





def analyze_ghsom_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),
        html.Hr(),
        html.Div(children=[ 

            #Card Mapa neurona winners
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_ghsom_1"))
                ),
                dbc.Collapse(id="collapse_ghsom_1",children=
                    dbc.CardBody(children=[ 
                        html.Div(id = 'winners_map_ghsom',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="ver_winners_map_ghsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),

                        html.Div(id = 'winners_submap_ghsom',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                        ),
                    ]),
                ),
            ]),


            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de componentes",color="link",id="button_collapse_ghsom_2"))
                ),
                dbc.Collapse(id="collapse_ghsom_2",children=
                    dbc.CardBody(children=[
                        dbc.CardBody(children=[
                            html.H5("Seleccionar atributos para mostar:"),
                            dcc.Dropdown(
                                id='dropdown_atrib_names_ghsom',
                                options=session_data.get_nombres_atributos(),
                                multi=True
                            ),
                            html.Div( 
                                [dbc.Checklist(
                                    options=[{"label": "Seleccionar todos", "value": 1}],
                                    value=[],
                                    id="check_seleccionar_todos_mapas_ghsom"),
                                dbc.Button("Ver Mapas de Componentes", id="ver_mapas_componentes_button_ghsom", className="mr-2", color="primary")],
                                style={'textAlign': 'center'}
                            ),
                            html.Div(id='component_plans_figures_ghsom_div', children=[''],
                                    style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                            )

                        ]),
                    ]),
                ),
            ]),



            #Card: U Matrix
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Matriz U",color="link",id="button_collapse_ghsom_3"))
                ),
                dbc.Collapse(id="collapse_ghsom_3",children=
                    dbc.CardBody(children=[

                        html.Div(id = 'umatrix_div_fig_ghsom',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="ver_umatrix_ghsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )



                    ])

                    ),
            ]),


            #Card: Guardar modelo
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Guardar modelo entrenado",color="link",id="button_collapse_ghsom_4"))
                ),
                dbc.Collapse(id="collapse_ghsom_4",children=
                    dbc.CardBody(children=[
                  
                        html.Div(children=[
                            
                            html.H5("Nombre del fichero"),
                            dbc.Input(id='nombre_de_fichero_a_guardar_ghsom',placeholder="Nombre del archivo", className="mb-3"),

                            dbc.Button("Guardar modelo", id="save_model_ghsom", className="mr-2", color="primary"),
                            html.P('',id="check_correctly_saved_ghsom")
                            ],
                            style={'textAlign': 'center'}
                        ),
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
#                       AUX FUNCTIONS
##################################################################


def get_gsom_plotted_graph(figure_id,name, gsom,mapped_dataset):
    '''
        Aux fun to plot gsom to heatmaps
    
        Args:
            figure_id(string):
            gsom(gsom):gsom to plot
            mapped_data(data): data mappaed to the gsom through the neuron parent of the gsom
        Returns:
            figure(dcc.Graph)
    '''

    #TODO add annotations arg to this fun

    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()

    
 



    #visualizacion

    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    positions={}

    # Getting winnig neurons for each data element
    for i,d in enumerate(mapped_dataset):
        winner_neuron = gsom.winner_neuron(d)[0][0]
        r, c = winner_neuron.position
        if((r,c) in positions):
            positions[(r,c)].append(mapped_dataset[i][-1]) 
        else:
            positions[(r,c)] = []
            positions[(r,c)].append(mapped_dataset[i][-1]) 


    # Obtener clases representativas de cada neurona
    for i in range(tam_eje_vertical):
        for j in range(tam_eje_horizontal):
            if((i,j) in positions):

                if(session_data.get_discrete_data() ):        #showing the class more represented in each neuron
                    c = Counter(positions[(i,j)])
                    data_to_plot[i][j] = c.most_common(1)[0][0]
                else: #continuos data: mean of the mapped values in each neuron
                    data_to_plot[i][j]  = np.mean(positions[(i,j)])
    
            else:
                data_to_plot[i][j] = None

   
        

    
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
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }   
    layout['width'] = 500
    layout['height']= 500
    layout['title'] = name
    #layout['x'] = 'Longitud(neuronas) del GSOM'
    #layout['y'] = 'Longitud(neuronas) del GSOM'

    annotations = []
    fig = dict(data=data, layout=layout)
    return dcc.Graph(id=figure_id,figure=fig)
    








##################################################################
#                       CALLBACKS
##################################################################

@app.callback(
    [Output(f"collapse_ghsom_{i}", "is_open") for i in range(1, 5)],
    [Input(f"button_collapse_ghsom_{i}", "n_clicks") for i in range(1, 5)],
    [State(f"collapse_ghsom_{i}", "is_open") for i in range(1, 5)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4, is_open1, is_open2,is_open3,is_open4):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_ghsom_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4
    elif button_id == "button_collapse_ghsom_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4
    elif button_id == "button_collapse_ghsom_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4
    elif button_id == "button_collapse_ghsom_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4
    return False, False, False,False





#Winners map
@app.callback(Output('winners_map_ghsom','children'),
              Input('ver_winners_map_ghsom_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_winner_map_ghsom(click):


    params = session_data.get_ghsom_model_info_dict()
    zero_unit = session_data.get_modelo()
    ghsom_nivel_1 = zero_unit.child_map
    tam_eje_vertical,tam_eje_horizontal=  ghsom_nivel_1.map_shape()
    print('Las nuevas dimensiones del mapa NIVEL 1 son(vertical,horizontal):',tam_eje_vertical,tam_eje_horizontal)


    dataset = session_data.get_data()
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]


   
    

    #visualizacion

    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    positions={}

    # Getting winnig neurons for each data element
    for i,d in enumerate(data):
        winner_neuron = ghsom_nivel_1.winner_neuron(d)[0][0]
        r, c = winner_neuron.position
        if((r,c) in positions):
            positions[(r,c)].append(dataset[i][-1]) 
        else:
            positions[(r,c)] = []
            positions[(r,c)].append(dataset[i][-1]) 

    #print('posiciones:', positions)

    # Obtener clases representativas de cada neurona
    for i in range(tam_eje_vertical):
        for j in range(tam_eje_horizontal):
            if((i,j) in positions):

                if(session_data.get_discrete_data() ):        #showing the class more represented in each neuron
                    c = Counter(positions[(i,j)])
                    data_to_plot[i][j] = c.most_common(1)[0][0]
                else: #continuos data: mean of the mapped values in each neuron
                    data_to_plot[i][j]  = np.mean(positions[(i,j)])
    
            else:
                data_to_plot[i][j] = None

   
        

    #print('data_to_plot:',data_to_plot)
    
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
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
    layout['width'] = 700
    layout['height']= 700
    layout['title'] = 'Mapa de neuronas ganadoras'
    annotations = []
    fig = dict(data=data, layout=layout)


    children = [ dcc.Graph(id='winnersmap_fig_ghsom',figure=fig)  ]

    print('\nVISUALIZACION:ghsom renderfinalizado\n')


    return children



#View submaps
@app.callback(Output('winners_submap_ghsom','children'),
              Input('winnersmap_fig_ghsom','clickData'),
              State('winners_submap_ghsom','children'),
              prevent_initial_call=True 
              )
def view_submaps(clickdata, plots_list):

    if(clickdata is None):
        return plots_list

    print('clikedpoint:',clickdata)
    #{'points': [{'curveNumber': 0, 'x': 0, 'y': 0, 'z': 0}]}
    points = clickdata['points']
    punto_clickeado = points[0]
    cord_vertical_punto_clickeado = punto_clickeado['x']
    cord_horizontal_punto_clickeado = punto_clickeado['y'] 
    
    #z_data_punto_clickeado = punto_clickeado['z']


 


    zero_unit = session_data.get_modelo()
    gsom_nivel_1 = zero_unit.child_map
    neuronas = gsom_nivel_1.get_neurons()
    #neuronas[(row,col)]
    selected_neuron = neuronas[(cord_horizontal_punto_clickeado,cord_vertical_punto_clickeado)]
    selected_gsom = selected_neuron.child_map
    if(selected_gsom is None):
        #return ['Esta neurona no tiene ningún submapa']
        return plots_list



    id = 'gsom_level_2_son_of_' + str(cord_vertical_punto_clickeado) + '_' + str(cord_horizontal_punto_clickeado)
    name = 'Nivel 2: GSOM hijo de la neurona (' + str(cord_vertical_punto_clickeado) + ',' + str(cord_horizontal_punto_clickeado) + ') del Nivel 1.'
    graph_gsom = get_gsom_plotted_graph(id ,name, selected_gsom,zero_unit.input_dataset)



    print('\nVISUALIZACION:ghsom nivel 2  renderfinalizado\n')
    if(len(plots_list) == 0):
        children = [ graph_gsom ]
        return children
    else:
        plots_list.append(graph_gsom)
        return plots_list


  










#Save file name
@app.callback(Output('nombre_de_fichero_a_guardar_ghsom', 'valid'),
              Output('nombre_de_fichero_a_guardar_ghsom', 'invalid'),
              Input('nombre_de_fichero_a_guardar_ghsom', 'value'),
              prevent_initial_call=True
              )
def check_savemodel_name(value):
    
    if not normpath(value) or search(r'[^A-Za-z0-9_\-]',value):
        return False,True
    else:
        return True,False


#Save ghsom model
@app.callback(Output('check_correctly_saved_ghsom', 'children'),
              Input('save_model_ghsom', 'n_clicks'),
              State('nombre_de_fichero_a_guardar_ghsom', 'value'),
              State('nombre_de_fichero_a_guardar_ghsom', 'valid'),
              prevent_initial_call=True )
def save_ghsom_model(n_clicks,name,isvalid):

    if(not isvalid):
        return ''

    data = []

    params = session_data.get_ghsom_model_info_dict()

    data.append('ghsom')
    data.append(params)
    data.append(session_data.get_modelo())

    '''
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M")
    filename = 'ghsom_model_' + dt_string + '.pickle'
    '''
    filename =   name +  '_ghsom.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Modelo guardado correctamente. Nombre del fichero: ' + filename