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

import plotly.graph_objects as go

from  views.session_data import session_data
from  config.config import *
import pickle

from  os.path import normpath 
from re import search 

import networkx as nx



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

                        

                        html.Div(id = 'grafo_ghsom',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
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
            figure(plotly)
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
    #layout['title'] = name
    #layout['x'] = 'Longitud(neuronas) del GSOM'
    #layout['y'] = 'Longitud(neuronas) del GSOM'

    annotations = []
    fig = dict(data=data, layout=layout)
    return fig
    


#Plot fig eith titles and gso size
def get_figure_complete_div(fig,fig_id, title,gsom_level,tam_eje_horizontal, tam_eje_vertical,neurona_padre):
    '''

        neurona_padre: None or str tuple if it exits
    '''

    
    if(neurona_padre is not None):
        div_info_neurona_padre = html.Div(children = [
            dbc.Badge('Neurona padre:', pill=True, color="light", className="mr-1"),
            dbc.Badge(neurona_padre, pill=True, color="info", className="mr-1")
        ])
       
    else:
        div_info_neurona_padre= ''




    div_inf_grid = html.Div(children = [
        html.H3(title),

        html.Div(children= [
            dbc.Badge('Nivel '+ str(gsom_level), pill=True , color="info", className="mr-1"),
            div_info_neurona_padre
        ], style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-direction': 'column '}),

        html.Div(children= [
            dbc.Badge(tam_eje_horizontal, pill=True, color="info", className="mr-1"),
            dbc.Badge('x', pill=True, color="light", className="mr-1"),
            dbc.Badge(tam_eje_vertical, pill=True, color="info", className="mr-1"),
            dbc.Badge('neuronas.', pill=True, color="light", className="mr-1")
        ], style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'})
        
    ], style={'margin': '0 auto','width': '100%', 'display': 'flex','align-items': 'center', 'justify-content': 'center',
                'flex-wrap': 'wrap', 'flex-direction': 'column ' })


      
    children =[ div_inf_grid, dcc.Graph(id=fig_id,figure=fig)  ]
    div = html.Div(children=children, style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                             'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } )

    return div





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
    #layout['title'] = 'Mapa de neuronas ganadoras'
    annotations = []
    fig = dict(data=data, layout=layout)


    div = get_figure_complete_div(fig,'winnersmap_fig_ghsom','Mapa de neuronas ganadoras',1,tam_eje_horizontal, tam_eje_vertical,None)

    print('\nVISUALIZACION:ghsom renderfinalizado\n')

    return div






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
    tam_eje_vertical,tam_eje_horizontal=  selected_gsom.map_shape()

    if(selected_gsom is None):
        #return ['Esta neurona no tiene ningún submapa']
        return plots_list



    id = 'gsom_nivel_2_son_of_' + str(cord_vertical_punto_clickeado) + '_' + str(cord_horizontal_punto_clickeado)
    name = 'Mapa de neuronas ganadoras'
    fig = get_gsom_plotted_graph(id ,name, selected_gsom,zero_unit.input_dataset)
    coordenada_neurona_padre = '(' + str(cord_vertical_punto_clickeado) + ',' + str(cord_horizontal_punto_clickeado) + ')'
    fig_in_div = get_figure_complete_div(fig,id, name,2 ,tam_eje_horizontal, tam_eje_vertical,coordenada_neurona_padre)



    print('\nVISUALIZACION:ghsom nivel 2  renderfinalizado\n')
    if(len(plots_list) == 0):
        children = [ fig_in_div ]
        return children
    else:
        plots_list.append(fig_in_div)
        return plots_list








#grafo map
@app.callback(Output('grafo_ghsom','children'),
              Input('ver_winners_map_ghsom_button','n_clicks'),
              prevent_initial_call=True 
              )
def ver_grafo_gsom(click):

    zero_unit = session_data.get_modelo()
    #g = zero_unit.graph
    grafo = nx.Graph()
    g = zero_unit.child_map.get_structure_graph(grafo,level=0)

    
    for n1,n2,attr in g.edges(data=True):
        print ('a verrr',n1,n2,attr)
    

    edge_x = []
    edge_y = []
    edge_z = []

    n= len(g.nodes)

    nodes_dict = {}
    counter = 0
    for node in g.nodes:
        if(node not in nodes_dict):
            nodes_dict[node] = counter
            counter = counter + 1

    


    for edge in g.edges:
        nodo1,nodo2 = edge
        a= nodes_dict[nodo1]
        b=nodes_dict[nodo2]
        '''
        edge_x.append(0)
        edge_y.append(g.nodes[nodo1]['nivel'] )
        edge_z.append(a)

        edge_x.append(0)
        edge_y.append(g.nodes[nodo2]['nivel'])
        edge_z.append(b)
        '''
        #2d
        edge_x.append(a)
        edge_y.append(-g.nodes[nodo1]['nivel'] )

        edge_x.append(b)
        edge_y.append(-g.nodes[nodo2]['nivel'])
        
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        #z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_z = []
    for node in g.nodes:
        '''
        node_x.append(0)
        node_y.append( g.nodes[node]['nivel'] )
        node_z.append(nodes_dict[node])
        '''

        node_x.append(nodes_dict[node])
        node_y.append(- g.nodes[node]['nivel'] )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        #z = node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    '''
    #Color nodes
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(g.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    '''


    data1=[edge_trace, node_trace]

    axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

    layout = go.Layout(
            title="Estructura de los submapas que componen la red",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
           


    fig = dict(data=data1,layout=layout)
    #fig = go.Figure(data=data1, layout=layout)


    children =[ dcc.Graph(id='grafo_ghsom',figure=fig)  ]
    return html.Div(children=children, style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                             'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } )



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