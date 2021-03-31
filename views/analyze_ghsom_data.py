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
                    
                    
                        html.Div(id = 'grafo_ghsom_1',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                        ),

                        html.Div(   id = 'winners_map_ghsom',children = '',
                                    style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                            'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } 
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
                                    id="check_seleccionar_todos_mapas_ghsom")],
                                style={'textAlign': 'center'}
                            ),

                            html.Div(id = 'grafo_ghsom_2',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
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

                        html.Div(id = 'grafo_ghsom_3',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                        ),

                        html.Div(id = 'umatrix_div_fig_ghsom',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
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


def get_gsom_plotted_graph(figure_id,name, gsom,neurons_mapped_targets):
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

    # Obtener clases representativas de cada neurona
    for i in range(tam_eje_vertical):
        for j in range(tam_eje_horizontal):

            if((i,j) in neurons_mapped_targets):

                if(session_data.get_discrete_data() ):        #showing the class more represented in each neuron
                    c = Counter(neurons_mapped_targets[(i,j)])
                    data_to_plot[i][j] = c.most_common(1)[0][0]
                else: #continuos data: mean of the mapped values in each neuron
                    data_to_plot[i][j]  = np.mean(neurons_mapped_targets[(i,j)])
    
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
    


#Plot fig eight titles and gso size
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
    '''
    div = html.Div(children=children, style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                             'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } )
    '''
    return children






# Grafo de la estructura del ghsom
# This 2 funcs are splitted in 2 for eficience. reason
def get_ghsom_graph_div(fig,dcc_graph_id):
    children =[ dcc.Graph(id=dcc_graph_id,figure=fig)  ]

    div =  html.Div(children=children, style={'margin': '0 auto','width': '100%', 'display': 'flex',
                                             'align-items': 'center', 'justify-content': 'center',
                                            'flex-wrap': 'wrap', 'flex-direction': 'column ' } )

    return div

# Grafo de la estructura del ghsom
def get_ghsom_fig():
    zero_unit = session_data.get_modelo()
    #TODO BORRAR ESTO EN GHSOM
    #g = zero_unit.graph
    grafo = nx.Graph()
    dataset = session_data.get_dataset()
    g = zero_unit.child_map.get_structure_graph(grafo,dataset ,level=0)
    session_data.set_ghsom_structure_graph(g)
  
    edge_x = []
    edge_y = []


    nodes_dict = {}
    counter = 0
    for node in g.nodes:
        nodes_dict[node] = counter
        counter +=  1


    for edge in g.edges:
        nodo1,nodo2 = edge
        a= nodes_dict[nodo1]
        b=nodes_dict[nodo2]
     
        edge_x.append(a)
        edge_y.append(-g.nodes[nodo1]['nivel'] )

        edge_x.append(b)
        edge_y.append(-g.nodes[nodo2]['nivel'])

        #Fix to plot segments
        edge_x.append(None)
        edge_y.append(None)


    #TODO quicker scattergl
    edge_trace = go.Scattergl(
    #edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        #z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    hover_text= []
    nodes_by_cords_dict = {}

    for node in g.nodes:

        x_cord = nodes_dict[node]
        y_cord = - g.nodes[node]['nivel'] 
        node_x.append(x_cord)
        node_y.append(y_cord)
        nodes_by_cords_dict[(y_cord, x_cord)] = node

        #Coordenadas de la neurona padre
        if('neurona_padre_pos' in g.nodes[node] ):
            cord_ver,cord_hor = g.nodes[node]['neurona_padre_pos']
            string = '(' + str(cord_hor) + ','+ str(cord_ver) + ')'
            hover_text.append(string) 
        else:
            hover_text.append('')

        


    session_data.set_ghsom_nodes_by_coord_dict(nodes_by_cords_dict)


    #node_trace = go.Scatter(
    #TODO quicker scattergl
    node_trace = go.Scattergl(
        x=node_x, 
        y=node_y,
        mode='markers',
        text=hover_text,
        hovertemplate= 'Coordenadas neurona padre: <b>%{text}</b><br>'+
                        'Nivel: %{y}<br>' 
                        +"<extra></extra>"
        ,
        marker=dict(
            color=['blue'], #set color equal to a variable
            size=14)
    )


    #TODO: esto es por esttetica del color, si va muy lento borrar
    '''
    node_adjacencies = []
    for node, adjacencies in enumerate(g.adjacency() ) :
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    '''

    data1=[edge_trace, node_trace]


    layout = go.Layout(
            title="Estructura de los submapas que componen la red",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
           
    fig = dict(data=data1,layout=layout)
    return fig





#For calculating u-matrix distiances

#Aux fun
def get_distances(weights_map, saved_distances, x,y,a,b):
    '''
        Aux. fun for ver_umatrix_gsom_fig callbacks to optimize the calc. of umatrix
    '''
    if (  (x,y,a,b) in saved_distances ):
        return saved_distances[(x,y,a,b)]
    elif (  (a,b,x,y) in saved_distances):
        return saved_distances[(a,b,x,y)]
    else:
        v1 = weights_map[(x,y)]
        v2 = weights_map[(a,b)]
        distancia = np.linalg.norm(v1-v2) # euclidean distance
        saved_distances[(x,y,a,b)] = distancia
        return distancia





##################################################################
#                       CALLBACKS
##################################################################
#Control de pliegues y carga del grafo de la estructura del ghsom
@app.callback(
    [Output(f"collapse_ghsom_{i}", "is_open") for i in range(1, 5)],
    Output('grafo_ghsom_1','children'),
    Output('grafo_ghsom_2','children'),
    Output('grafo_ghsom_3','children'),
    [Input(f"button_collapse_ghsom_{i}", "n_clicks") for i in range(1, 5)],
    [State(f"collapse_ghsom_{i}", "is_open") for i in range(1, 5)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4, is_open1, is_open2,is_open3,is_open4):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False, [],[],[]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        fig=  get_ghsom_fig()
        div_1 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_1')
        div_2 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_2')
        div_3 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_3')

        
    if button_id == "button_collapse_ghsom_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4, div_1,div_2,div_3
    elif button_id == "button_collapse_ghsom_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4, div_1,div_2,div_3
    elif button_id == "button_collapse_ghsom_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4, div_1,div_2,div_3
    elif button_id == "button_collapse_ghsom_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4, div_1,div_2,div_3
    return False, False, False,False, div_1,div_2,div_3






#Winners map del punto seleccionado del grafo
@app.callback(Output('winners_map_ghsom','children'),
              Input('dcc_ghsom_graph_1','clickData'),
              prevent_initial_call=True 
              )
def view_winner_map_by_selected_point(clickdata):

    if(clickdata is None):
        raise PreventUpdate

    print('clikedpoint:',clickdata)
    #{'points': [{'curveNumber': 0, 'x': 0, 'y': 0, 'z': 0}]}
    points = clickdata['points']
    punto_clickeado = points[0]
    cord_horizontal_punto_clickeado = punto_clickeado['x']
    cord_vertical_punto_clickeado = punto_clickeado['y'] 
    
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    ghsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    tam_eje_vertical,tam_eje_horizontal=  ghsom.map_shape()

    g = session_data.get_ghsom_structure_graph()
    neurons_mapped_targets = g.nodes[ghsom]['neurons_mapped_targets']
    level = g.nodes[ghsom]['nivel']

    #visualizacion
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    positions={}

    #solo para el input dataset tiene que serrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr


    # Obtener clases representativas de cada neurona
    for i in range(tam_eje_vertical):
        for j in range(tam_eje_horizontal):

            if((i,j) in neurons_mapped_targets):

                if(session_data.get_discrete_data() ):        #showing the class more represented in each neuron
                    c = Counter(neurons_mapped_targets[(i,j)])
                    data_to_plot[i][j] = c.most_common(1)[0][0]
                else: #continuos data: mean of the mapped values in each neuron
                    data_to_plot[i][j]  = np.mean(neurons_mapped_targets[(i,j)])
    
            else:
                data_to_plot[i][j] = None

   
        

    
    #type= heatmap para mas precision
    #heatmapgl
    trace = dict(type='heatmap', z=data_to_plot, colorscale = 'Jet')
    data=[trace]

    
    data.append({'type': 'scattergl',
                    'mode': 'text',
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

    children = get_figure_complete_div(fig,'winnersmap_fig_ghsom','Mapa de neuronas ganadoras',level,tam_eje_horizontal, tam_eje_vertical,None)

    print('\nVISUALIZACION:ghsom renderfinalizado\n')

    return children




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











# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names_ghsom','value'),
    Input("check_seleccionar_todos_mapas_ghsom", "value"),
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



#Actualizar mapas de componentes
@app.callback(Output('component_plans_figures_ghsom_div','children'),
              Input('dcc_ghsom_graph_2','clickData'),
              State('dropdown_atrib_names_ghsom','value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_ghsom_fig(clickdata,names):


    if(clickdata is None):
        raise PreventUpdate

    print('clikedpoint:',clickdata)
    #{'points': [{'curveNumber': 0, 'x': 0, 'y': 0, 'z': 0}]}
    points = clickdata['points']
    punto_clickeado = points[0]
    cord_horizontal_punto_clickeado = punto_clickeado['x']
    cord_vertical_punto_clickeado = punto_clickeado['y'] 
    
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    gsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()


    
    #Weights MAP
    weights_map= gsom.get_weights_map()
    # weights_map[(row,col)] = np vector whith shape=n_feauters, dtype=np.float32




    # Getting selected attrribs indexes
    with open(SESSION_DATA_FILE_DIR) as json_file:
        datos_entrenamiento = json.load(json_file)

    nombres_columnas = datos_entrenamiento['columns_names']
    nombres_atributos = nombres_columnas[0:len(nombres_columnas)-1]
    lista_de_indices = []

   

    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    

    traces = []


    xaxis_dict ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    yaxis_dict  ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }

    for k in lista_de_indices:
        data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                data_to_plot[i][j] = weights_map[(i,j)][k]
        
      
       

        figure= go.Figure(layout= {"height": 300,'width' : 300, 'title': nombres_atributos[k], 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict },
                          data=go.Heatmap(z=data_to_plot,showscale= True)                                                      
        ) 

        id ='graph-{}'.format(k)

        traces.append(
            html.Div(children= dcc.Graph(id=id,figure=figure)
            ) 
        )
                   
    return traces






#Ver UMatrix GHSOM
@app.callback(Output('umatrix_div_fig_ghsom','children'),
              Input('dcc_ghsom_graph_3','clickData'),
              prevent_initial_call=True 
              )
def ver_umatrix_gsom_fig(clickdata):


    if(clickdata is None):
        raise PreventUpdate

    print('clikedpoint:',clickdata)
    #{'points': [{'curveNumber': 0, 'x': 0, 'y': 0, 'z': 0}]}
    points = clickdata['points']
    punto_clickeado = points[0]
    cord_horizontal_punto_clickeado = punto_clickeado['x']
    cord_vertical_punto_clickeado = punto_clickeado['y'] 
    
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    gsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()

    

    #Weights MAP
    weights_map= gsom.get_weights_map()
    # weights_map[(row,col)] = np vector whith shape=n_feauters, dtype=np.float32


    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)

    saved_distances= {} #for saving distances
    # saved_distances[i,j,a,b] with (i,j) and (a,b) neuron cords

    '''
    debugg
    for i in range(tam_eje_vertical):
        for j in range(tam_eje_horizontal):
            print('pesos[][]: ',i, j, '---:  ',weights_map[(i,j)])
    print('eje x e y: ',tam_eje_vertical,tam_eje_horizontal)

    '''
    for i in range(tam_eje_vertical):
        for j in range(tam_eje_horizontal):

            neuron_neighbords = []
           
            
            if(j-1 >= 0): #bottom   neighbor
                neuron_neighbords.append( get_distances(weights_map, saved_distances, i,j,i,j-1))
            if(j+1 < tam_eje_horizontal):#top  neighbor
                neuron_neighbords.append( get_distances(weights_map, saved_distances, i,j,i,j+1))
            if(i-1 >= 0): #  #left  neighbor
                neuron_neighbords.append( get_distances(weights_map, saved_distances, i,j,i-1,j))
            if(i+1 < tam_eje_vertical ): #right neighbor
                neuron_neighbords.append( get_distances(weights_map, saved_distances, i,j,i+1,j))

            if(any(neuron_neighbords) ):
                data_to_plot[i][j] = sum(neuron_neighbords)/len(neuron_neighbords)

    '''
    debug
    print('distancias' )
    for item in saved_distances.items():
        print(item)
    '''
    trace = dict(type='heatmap', z=data_to_plot, colorscale = 'Jet')
    data=[trace]

    
    data.append({'type': 'scattergl',
                    'mode': 'text',
                    'text': 'a'
                    })
    
    layout = {}
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }  
    layout['width'] = 700
    layout['height']= 700
    annotations = []
    fig = dict(data=data, layout=layout)


    children = [ dcc.Graph(id='umatrix_fig_ghsom',figure=fig)  ]

    print('\nVISUALIZACION:gsom renderfinalizado\n')



    return children