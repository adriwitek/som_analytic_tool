import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
import dash
import  views.elements as elements
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil
import numpy as np
from collections import Counter
from datetime import datetime


from  views.session_data import session_data
from  config.config import DIR_SAVED_MODELS


import pickle

from  os.path import normpath 
from re import search 
import views.plot_utils as pu



def analyze_gsom_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),

        html.H6('Parámetros de entrenamiento',className="card-title"  ),
        html.Div(id = 'info_table_gsom',children=info_trained_params_gsom_table(),style={'textAlign': 'center'} ),

        html.Div(children=[ 
            
            #Card Estadísticas
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Estadísticas",color="link",id="button_collapse_gsom_1"),style={'textAlign': 'center', 'justify':'center'})
                ),
                dbc.Collapse(id="collapse_gsom_1",children=
                    dbc.CardBody(children=[ 
                        html.Div( id='div_estadisticas_gsom',children = '', style={'textAlign': 'center'}),
                        html.Div([
                            dbc.Button("Ver", id="ver_estadisticas_gsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),

            #Card Mapa neurona winners
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_gsom_2"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_gsom_2",children=
                    dbc.CardBody(children=[ 
                        html.Div(id = 'div_winners_map_gsom',children='',
                                style= pu.get_single_heatmap_css_style()
                        ),
                        html.Div([  
                                dbc.Checklist(options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                            value=[],
                                            id="check_annotations_win_gsom"),
                                dbc.Button("Ver", id="ver_winners_map_gsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),



            #Card Mapa frecuencias
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de frecuencias",color="link",id="button_collapse_gsom_3"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_gsom_3",children=
                    dbc.CardBody(children=[ 
                        html.Div(id = 'div_freq_map_gsom',children='',
                                style= pu.get_single_heatmap_css_style()
                        ),
                        html.Div([  
                                dbc.Button("Ver", id="ver_freq_map_gsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),


            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de componentes",color="link",id="button_collapse_gsom_4"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_gsom_4",children=
                    dbc.CardBody(children=[
                        dbc.CardBody(children=[
                            html.H5("Seleccionar atributos para mostar:"),
                            dcc.Dropdown(
                                id='dropdown_atrib_names_gsom',
                                options=session_data.get_dataset_atrib_names_dcc_dropdown_format(),
                                multi=True
                            ),
                            html.Div( 
                                [dbc.Checklist(
                                    options=[{"label": "Seleccionar todos", "value": 1}],
                                    value=[],
                                    id="check_seleccionar_todos_mapas_gsom"),
                                dbc.Checklist(  options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                                value=[],
                                                id="check_annotations_comp_gsom"),
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
                    html.H2(dbc.Button("Matriz U",color="link",id="button_collapse_gsom_5"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_gsom_5",children=
                    dbc.CardBody(children=[

                        html.Div(id = 'umatrix_div_fig_gsom',children = '',
                                style= pu.get_single_heatmap_css_style()
                        ),
                        html.Div( 
                            [dbc.Checklist(  options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                                value=[],
                                                id="check_annotations_umax_gsom"),
                            dbc.Button("Ver", id="ver_umatrix_gsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )



                    ])

                    ),
            ]),


            #Card: Guardar modelo
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Guardar modelo entrenado",color="link",id="button_collapse_gsom_6"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_gsom_6",children=
                    dbc.CardBody(children=[
                  
                        html.Div(children=[
                            
                            html.H5("Nombre del fichero"),
                            dbc.Input(id='nombre_de_fichero_a_guardar',placeholder="Nombre del archivo", className="mb-3"),

                            dbc.Button("Guardar modelo", id="save_model_gsom", className="mr-2", color="primary"),
                            html.P('',id="check_correctly_saved_gsom")
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

def info_trained_params_gsom_table():

    info = session_data.get_gsom_model_info_dict()

    
    #Table
    table_header = [
         html.Thead(html.Tr([
                        html.Th("Tam. Horizontal Inicial del Grid"),
                        html.Th("Tam. Vertical Inicial del Grid"),
                        html.Th("Tau 1"),
                        html.Th("Tasa Aprendizaje"),
                        html.Th("Decadencia"),
                        html.Th("Sigma Gaussiana"),
                        html.Th("Épocas por iteracción"),
                        html.Th("Num. Máximo de Iteraciones"),
                        html.Th("Semilla")
        ]))
    ]



    if(info['check_semilla'] == 0):
        semilla = 'No'
    else:
        semilla = 'Sí: ' + str(info['seed']) 
   
    row_1 = html.Tr([html.Td( info['tam_eje_horizontal']),
                    html.Td( info['tam_eje_vertical']),
                     html.Td( info['tau_1']) ,
                     html.Td( info['learning_rate']),
                     html.Td( info['decadency']) ,
                     html.Td( info['sigma']) ,
                     html.Td( info['epocas_gsom']) ,
                     html.Td( info['max_iter_gsom']  ),
                     html.Td( semilla )

    ]) 

    table_body = [html.Tbody([row_1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children





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

@app.callback(
    [Output(f"collapse_gsom_{i}", "is_open") for i in range(1, 7)],
    [Input(f"button_collapse_gsom_{i}", "n_clicks") for i in range(1, 7)],
    [State(f"collapse_gsom_{i}", "is_open") for i in range(1, 7)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4,n5,n6, is_open1, is_open2,is_open3,is_open4,is_open5,is_open6):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_gsom_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4, is_open5, is_open6
    elif button_id == "button_collapse_gsom_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4, is_open5, is_open6
    elif button_id == "button_collapse_gsom_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4,is_open5, is_open6
    elif button_id == "button_collapse_gsom_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4, is_open5, is_open6
    elif button_id == "button_collapse_gsom_5" and n5:
        return is_open1, is_open2, is_open3,is_open4, not is_open5, is_open6
    elif button_id == "button_collapse_gsom_6" and n6:
        return is_open1, is_open2, is_open3,is_open4, is_open5, not is_open6
    return False, False, False,False,False,False



#Estadisticas
@app.callback(Output('div_estadisticas_gsom', 'children'),
              Input('ver_estadisticas_gsom_button', 'n_clicks'),
              prevent_initial_call=True )
def ver_estadisticas_gsom(n_clicks):

    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    qe, mqe = gsom.get_map_qe_and_mqe()

    #Table
    table_header = [
        html.Thead(html.Tr([html.Th("Magnitud"), html.Th("Valor")]))
    ]
    row0 = html.Tr([html.Td("Error de Cuantización"), html.Td(qe)])
    row1 = html.Tr([html.Td("Error de Cuantización Medio"), html.Td(mqe)])
    table_body = [html.Tbody([row0,row1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children



#Winners map
@app.callback(Output('div_winners_map_gsom','children'),
              Input('ver_winners_map_gsom_button','n_clicks'),
              State('check_annotations_win_gsom','value'),
              prevent_initial_call=True 
              )
def update_winner_map_gsom(click,check_annotations):

    params = session_data.get_gsom_model_info_dict()
    #tam_eje_vertical = params['tam_eje_vertical']
    #tam_eje_horizontal = params['tam_eje_horizontal']
    dataset = session_data.get_dataset()
    data = session_data.get_data()
    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()

    

    #visualizacion
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
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


   #Discrete data: most common class
    if(session_data.get_discrete_data() ):     
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in positions):
                        c = Counter(positions[(i,j)])
                        data_to_plot[i][j] = c.most_common(1)[0][0]
                else:
                    data_to_plot[i][j] = np.nan
    else:#continuos data: mean of the mapped values in each neuron
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in positions):
                        data_to_plot[i][j]  = np.mean(positions[(i,j)])
                else:
                    data_to_plot[i][j] = np.nan
        

    fig = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = None)
    children = pu.get_fig_div_with_info(fig,'winners_map_gsom', 'Mapa de Neuronas Ganadoras',tam_eje_horizontal, tam_eje_vertical)

    return children
   




#Frequency map
@app.callback(Output('div_freq_map_gsom','children'),
              Input('ver_freq_map_gsom_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_freq_map_gsom(click):

    params = session_data.get_gsom_model_info_dict()
    
    #tam_eje_vertical = params['tam_eje_vertical']
    #tam_eje_horizontal = params['tam_eje_horizontal']
    data = session_data.get_data()
    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()

    

    #visualizacion
    data_to_plot = np.zeros([tam_eje_vertical ,tam_eje_horizontal],dtype=int)

    # Getting winnig neurons for each data element
    for i,d in enumerate(data):
        winner_neuron = gsom.winner_neuron(d)[0][0]
        r, c = winner_neuron.position
        data_to_plot[r][c] = data_to_plot[r][c] + 1
     

    fig = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,True, title = None)
    children = pu.get_fig_div_with_info(fig,'freq_map_gsom', 'Mapa de Frecuencias por Neurona',tam_eje_horizontal, tam_eje_vertical)

    return children





#Habilitar boton ver_mapas_componentes_button_gsom
@app.callback(Output('ver_mapas_componentes_button_gsom','disabled'),
              Input('dropdown_atrib_names_gsom','value')
            )
def enable_ver_mapas_componentes_button(values):
    if ( values ):
        return False
    else:
        return True




#Actualizar mapas de componentes
@app.callback(Output('component_plans_figures_gsom_div','children'),
              Input('ver_mapas_componentes_button_gsom','n_clicks'),
              State('dropdown_atrib_names_gsom','value'),
              State('check_annotations_comp_gsom','value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_gsom_fig(click,names, check_annotations):


    params = session_data.get_gsom_model_info_dict()
    #tam_eje_vertical = params['tam_eje_vertical']
    #tam_eje_horizontal = params['tam_eje_horizontal']
 
    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()

    
    #Weights MAP
    weights_map= gsom.get_weights_map()
    # weights_map[(row,col)] = np vector whith shape=n_feauters, dtype=np.float32

    nombres_atributos = session_data.get_dataset_atrib_names()
    lista_de_indices = []

    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    

    traces = []

    for k in lista_de_indices:
        data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                data_to_plot[i][j] = weights_map[(i,j)][k]
        
      
        id ='graph-{}'.format(k)
        figure = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = nombres_atributos[k])
        children = pu.get_fig_div_with_info(figure,id, '',None, None,gsom_level= None,neurona_padre=None)

        

        traces.append(
            html.Div(children= children
            ) 
        )
                   
    return traces
  



# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names_gsom','value'),
    Input("check_seleccionar_todos_mapas_gsom", "value"),
    prevent_initial_call=True
    )
def on_form_change(check):

    if(check):
        atribs = session_data.get_dataset_atrib_names()
        return atribs
    else:
        return []






#Ver UMatrix GSOM
@app.callback(Output('umatrix_div_fig_gsom','children'),
              Input('ver_umatrix_gsom_button','n_clicks'),
              State('check_annotations_umax_gsom','value'),
              prevent_initial_call=True 
              )
def ver_umatrix_gsom_fig(click, check_annotations):

    print('Button clicked, calculating umatrix')

    params = session_data.get_gsom_model_info_dict()
    #tam_eje_vertical = params['tam_eje_vertical']
    #tam_eje_horizontal = params['tam_eje_horizontal']


    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
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
    fig = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = None)
    children =  pu.get_fig_div_with_info(fig,'umatrix_fig_gsom', 'Matriz U',tam_eje_horizontal, tam_eje_vertical)

    return children




#Save file name
@app.callback(Output('nombre_de_fichero_a_guardar', 'valid'),
              Output('nombre_de_fichero_a_guardar', 'invalid'),
              Input('nombre_de_fichero_a_guardar', 'value'),
              prevent_initial_call=True
              )
def check_savemodel_name(value):
    
    if not normpath(value) or search(r'[^A-Za-z0-9_\-]',value):
        return False,True
    else:
        return True,False




#Save GSOM model
@app.callback(Output('check_correctly_saved_gsom', 'children'),
              Input('save_model_gsom', 'n_clicks'),
              State('nombre_de_fichero_a_guardar', 'value'),
              State('nombre_de_fichero_a_guardar', 'valid'),
              prevent_initial_call=True )
def save_gsom_model(n_clicks,name,isvalid):

    if(not isvalid):
        return ''

    data = []

    params = session_data.get_gsom_model_info_dict()

    data.append('gsom')
    data.append(params)
    data.append(session_data.get_modelo())

    filename =   name +  '_gsom.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Modelo guardado correctamente. Nombre del fichero: ' + filename