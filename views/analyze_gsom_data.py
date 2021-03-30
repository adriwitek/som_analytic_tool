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
from datetime import datetime


from  views.session_data import session_data
from  config.config import *
import pickle

from  os.path import normpath 
from re import search 


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
                                options=session_data.get_nombres_atributos(),
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
                    html.H2(dbc.Button("Matriz U",color="link",id="button_collapse_gsom_3"))
                ),
                dbc.Collapse(id="collapse_gsom_3",children=
                    dbc.CardBody(children=[

                        html.Div(id = 'umatrix_div_fig_gsom',children = '',
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
                        html.Div( 
                            [dbc.Button("Ver", id="ver_umatrix_gsom_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )



                    ])

                    ),
            ]),


            #Card: Guardar modelo
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Guardar modelo entrenado",color="link",id="button_collapse_gsom_4"))
                ),
                dbc.Collapse(id="collapse_gsom_4",children=
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









#Winners map
@app.callback(Output('winners_map_gsom','figure'),
              Input('ver_winners_map_gsom_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_winner_map_gsom(click):

    params = session_data.get_gsom_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']

    dataset = session_data.get_dataset()
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]


    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    

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
    layout['height']= 700
    annotations = []
    fig = dict(data=data, layout=layout)

    #Poner o no etiquetas...

    print('\nVISUALIZACION:gsom renderfinalizado\n')

    return fig
   







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
              prevent_initial_call=True 
              )
def update_mapa_componentes_gsom_fig(click,names):


    params = session_data.get_gsom_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']


    dataset = session_data.get_dataset()
    data = dataset[:,:-1]
    targets = dataset[:,-1:]
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]


    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    
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
                   
    print('render finalizado')
    return traces
  



# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names_gsom','value'),
    Input("check_seleccionar_todos_mapas_gsom", "value"),
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



#Ver UMatrix GSOM
@app.callback(Output('umatrix_div_fig_gsom','children'),
              Input('ver_umatrix_gsom_button','n_clicks'),
              prevent_initial_call=True 
              )
def ver_umatrix_gsom_fig(click):

    print('Button clicked, calculating umatrix')

    params = session_data.get_gsom_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']


    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    

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
                    #'x': x_ticks,
                    #'y': y_ticks,
                    'text': 'a'
                    })
    
    layout = {}
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }  
    layout['width'] = 700
    layout['height']= 700
    annotations = []
    fig = dict(data=data, layout=layout)


    
    children = [ dcc.Graph(id='umatrix_fig_gsom',figure=fig)  ]

    print('\nVISUALIZACION:gsom renderfinalizado\n')



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

    '''
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M")
    filename = 'gsom_model_' + dt_string + '.pickle'
    '''
    filename =   name +  '_gsom.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Modelo guardado correctamente. Nombre del fichero: ' + filename