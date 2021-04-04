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
from  config.config import SESSION_DATA_FILE_DIR,DEFAULT_HEATMAP_PX_HEIGHT, DEFAULT_HEATMAP_PX_WIDTH,DEFAULT_HEATMAP_COLORSCALE
import pickle

from  os.path import normpath 
from re import search 

import networkx as nx
import views.plot_utils as pu



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

                        html.Div(
                            dbc.Checklist(  options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                            value=[],
                                            id="check_annotations_winmap_ghsom"),
                            style={'textAlign': 'center'}
                        )
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
                            ),

                            html.Div(dbc.Checklist(  options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                                    value=[],
                                                    id="check_annotations_comp_ghsom"),
                                        style={'textAlign': 'center'}
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
                        ),

                        html.Div(dbc.Checklist(     options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                                    value=[],
                                                    id="check_annotations_um_ghsom"),
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
    #edge_trace = go.Scattergl(
    edge_trace = go.Scatter(
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


    node_trace = go.Scatter(
    #TODO quicker scattergl
    #TODO Scattergl no muestra el hovertext
    #node_trace = go.Scattergl(
        x=node_x, 
        y=node_y,
        mode='markers',
        text=hover_text,
        hovertemplate= 'Coordenadas neurona padre: <b>%{text}</b><br>'+
                        'Nivel: %{y}<br>' 
                        +"<extra></extra>"
        ,
        marker=dict(
            #color=['blue',], #set color equal to a variable
            color = 'slategray',
            size=14)
    )



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




#Aux fun for calculating u-matrix distiances
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
              Output('dcc_ghsom_graph_1','figure'),
              Input('dcc_ghsom_graph_1','clickData'),
              Input('check_annotations_winmap_ghsom','value'),
              State('dcc_ghsom_graph_1','figure'),
              prevent_initial_call=True 
              )
def view_winner_map_by_selected_point(clickdata,check_annotations,figure):


    if clickdata is  None:
        raise PreventUpdate

    #print('clikedpoint:',clickdata)
    #{'points': [{'curveNumber': 0, 'x': 0, 'y': 0, 'z': 0}]}
    points = clickdata['points']
    punto_clickeado = points[0]
    cord_horizontal_punto_clickeado = punto_clickeado['x']
    cord_vertical_punto_clickeado = punto_clickeado['y'] 

    #Actualizar  COLOR DEL punto seleccionado en el grafo
    data_g = []
    data_g.append(figure['data'][0])
    data_g.append(figure['data'][1])

    data_g.append( go.Scattergl(
        x=[cord_horizontal_punto_clickeado], 
        y=[cord_vertical_punto_clickeado],
        mode='markers',
        marker=dict(
            color = 'blue',
            size=14)
    ))
    figure['data'] = data_g
    
    
    #Mapa de neuronas ganadoras del gsom seleccionado en el grafo
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    ghsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    tam_eje_vertical,tam_eje_horizontal=  ghsom.map_shape()

    g = session_data.get_ghsom_structure_graph()
    neurons_mapped_targets = g.nodes[ghsom]['neurons_mapped_targets']
    level = g.nodes[ghsom]['nivel']
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    neurona_padre_string = None
    if('neurona_padre_pos' in g.nodes[ghsom] ):
            cord_ver,cord_hor = g.nodes[ghsom]['neurona_padre_pos']
            neurona_padre_string = '(' + str(cord_hor) + ','+ str(cord_ver) + ')'

    # Obtener clases representativas de cada neurona
    #Discrete data: most common class
    if(session_data.get_discrete_data() ):     
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in neurons_mapped_targets):
                        c = Counter(neurons_mapped_targets[(i,j)])
                        data_to_plot[i][j] = c.most_common(1)[0][0]
                else:
                    data_to_plot[i][j] = np.nan
    else:#continuos data: mean of the mapped values in each neuron
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in neurons_mapped_targets):
                        data_to_plot[i][j]  = np.mean(neurons_mapped_targets[(i,j)])
                else:
                    data_to_plot[i][j] = np.nan

    '''
    #heatmapgl
    trace = dict(type='heatmap', z=data_to_plot, colorscale = DEFAULT_HEATMAP_COLORSCALE)
    data=[trace]
    data.append({'type': 'scattergl',
                    'mode': 'text'
                })
    layout = {}
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
    layout['width'] = DEFAULT_HEATMAP_PX_WIDTH
    layout['height']= DEFAULT_HEATMAP_PX_HEIGHT
    #layout['title'] = 'Mapa de neuronas ganadoras'


    #Annotations
    if(check_annotations  ): 
        #Todo replace None values with NaN values
        data_to_plot_1 = np.array(data_to_plot, dtype=np.float64)
        annotations = pu.make_annotations(data_to_plot_1, colorscale = DEFAULT_HEATMAP_COLORSCALE, reversescale= False)
        layout['annotations'] = annotations

    fig = dict(data=data, layout=layout)
    '''
    fig =  pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,check_annotations, title = None)

    children = pu.get_fig_div_with_info(fig,'winnersmap_fig_ghsom','Mapa de neuronas ganadoras',tam_eje_horizontal, tam_eje_vertical,level,neurona_padre_string)




    print('\nVISUALIZACION:ghsom renderfinalizado\n')
    return children,figure




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
              Output('dcc_ghsom_graph_2','figure'),
              Input('dcc_ghsom_graph_2','clickData'),
              Input('check_annotations_comp_ghsom','value'),
              State('dcc_ghsom_graph_2','figure'),
              State('dropdown_atrib_names_ghsom','value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_ghsom_fig(clickdata,check_annotations,fig_grafo,names):


    if(clickdata is None):
        raise PreventUpdate

    #{'points': [{'curveNumber': 0, 'x': 0, 'y': 0, 'z': 0}]}
    points = clickdata['points']
    punto_clickeado = points[0]
    cord_horizontal_punto_clickeado = punto_clickeado['x']
    cord_vertical_punto_clickeado = punto_clickeado['y'] 

    #Actualizar  COLOR DEL punto seleccionado en el grafo
    data_g = []
    data_g.append(fig_grafo['data'][0])
    data_g.append(fig_grafo['data'][1])

    data_g.append( go.Scattergl(
        x=[cord_horizontal_punto_clickeado], 
        y=[cord_vertical_punto_clickeado],
        mode='markers',
        marker=dict(
            color = 'blue',
            size=14)
    ))
    fig_grafo['data'] = data_g
    

    #Mapa de componentes del gsom seleccionado en el grafo
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    gsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()
    weights_map= gsom.get_weights_map()
    # weights_map[(row,col)] = np vector whith shape=n_feauters, dtype=np.float32


    #TODO: quietar json de  Getting selected attrribs indexes
    with open(SESSION_DATA_FILE_DIR) as json_file:
        datos_entrenamiento = json.load(json_file)

    nombres_columnas = datos_entrenamiento['columns_names']
    nombres_atributos = nombres_columnas[0:len(nombres_columnas)-1]
    lista_de_indices = []
    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    


    traces = []
    

    for k in lista_de_indices:
        data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                data_to_plot[i][j] = weights_map[(i,j)][k]
        

        figure =  pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,check_annotations, title = nombres_atributos[k])

        '''
        layout = {'title': nombres_atributos[k], 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict }
        #Annotations
        if(check_annotations  ): 
            #Todo replace None values with NaN values
            data_to_plot_1 = np.array(data_to_plot, dtype=np.float64)
            annotations = pu.make_annotations(data_to_plot_1, colorscale = DEFAULT_HEATMAP_COLORSCALE, reversescale= False)
            layout['annotations'] = annotations

        figure= go.Figure(layout= layout,
                          data=go.Heatmap(z=data_to_plot,showscale= True)                                                      
        ) 

        '''
        id ='graph-{}'.format(k)
        traces.append(
            html.Div(children= dcc.Graph(id=id,figure=figure)
            ) 
        )


    return traces, fig_grafo




#Ver UMatrix GHSOM
@app.callback(Output('umatrix_div_fig_ghsom','children'),
              Output('dcc_ghsom_graph_3','figure'),
              Input('dcc_ghsom_graph_3','clickData'),
              Input('check_annotations_um_ghsom','value'),
              State('dcc_ghsom_graph_3','figure'),
              prevent_initial_call=True 
              )
def ver_umatrix_ghsom_fig(clickdata,check_annotations,fig_grafo):


    if(clickdata is None):
        raise PreventUpdate

    points = clickdata['points']
    punto_clickeado = points[0]
    cord_horizontal_punto_clickeado = punto_clickeado['x']
    cord_vertical_punto_clickeado = punto_clickeado['y'] 
    
    #Actualizar  COLOR DEL punto seleccionado en el grafo
    data_g = []
    data_g.append(fig_grafo['data'][0])
    data_g.append(fig_grafo['data'][1])

    data_g.append( go.Scattergl(
        x=[cord_horizontal_punto_clickeado], 
        y=[cord_vertical_punto_clickeado],
        mode='markers',
        marker=dict(
            color = 'blue',
            size=14)
    ))
    fig_grafo['data'] = data_g


    #umatrix del gsom seleccionado en el grafo
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    gsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()
    weights_map= gsom.get_weights_map()
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    saved_distances= {} #for saving distances
    # saved_distances[i,j,a,b] with (i,j) and (a,b) neuron cords

    #TODO borrar:
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

    #TODO BORRAR
    '''
    debug
    print('distancias' )
    for item in saved_distances.items():
        print(item)
    '''

    '''
    trace = dict(type='heatmap', z=data_to_plot, colorscale = DEFAULT_HEATMAP_COLORSCALE)
    data=[trace]
    data.append({'type': 'scattergl',
                    'mode': 'text'
                })
    layout = {}
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }  
    layout['width'] = DEFAULT_HEATMAP_PX_WIDTH
    layout['height']= DEFAULT_HEATMAP_PX_HEIGHT
    
    #Annotations
    if(check_annotations  ): 
        #Todo replace None values with NaN values
        data_to_plot_1 = np.array(data_to_plot, dtype=np.float64)
        annotations = pu.make_annotations(data_to_plot_1, colorscale = DEFAULT_HEATMAP_COLORSCALE, reversescale= False)
        layout['annotations'] = annotations

    fig = dict(data=data, layout=layout)
    '''
    fig = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,check_annotations, title = None)

    children = [ dcc.Graph(id='umatrix_fig_ghsom',figure=fig)  ]

    print('\nVISUALIZACION:gsom renderfinalizado\n')

    return children, fig_grafo





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

    filename =   name +  '_ghsom.pickle'
    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Modelo guardado correctamente. Nombre del fichero: ' + filename




