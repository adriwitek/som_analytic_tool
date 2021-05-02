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


from  views.session_data import session_data
from  config.config import *
from  config.config import  DIR_SAVED_MODELS, UMATRIX_HEATMAP_COLORSCALE
import pickle
from  os.path import normpath 
from re import search 
import views.plot_utils as pu







def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos \n',className="card-title"  ),

        html.H6('Parámetros de entrenamiento',className="card-title"  ),
        html.Div(id = 'info_table_som',children=info_trained_params_som_table(),style={'textAlign': 'center'} ),

        html.Div(children=[ 

    
            #Card Estadísticas
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Estadísticas",color="link",id="button_collapse_1"),style={'textAlign': 'center', 'justify':'center'})
                ),
                dbc.Collapse(id="collapse_1",children=
                    dbc.CardBody(children=[ 
                        html.Div( id='div_estadisticas_som',children = '', style={'textAlign': 'center'}),
                        html.Div([
                            dbc.Button("Ver", id="ver_estadisticas_som_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),

            #Card Mapa neurona winners
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_2"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_2",children=
                    dbc.CardBody(children=[ 
                        html.Div( id='div_mapa_neuronas_ganadoras',children = '', style= pu.get_single_heatmap_css_style()),
                        html.Div([
                            dbc.Checklist(  options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                            value=[],
                                            id="check_annotations_winnersmap"),
                            dbc.Button("Ver", id="ver", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ]),
                ),
            ]),



            #Card: Frecuencias de activacion
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de frecuencias de activación",color="link",id="button_collapse_3"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_3",children=
                    dbc.CardBody(children=[
                        html.Div( id='div_frequency_map',children = '',style= pu.get_single_heatmap_css_style()),
                        html.Div([ 
                            dbc.Checklist(options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                            value=[],
                                            id="check_annotations_freq"),
                                        
                            dbc.Button("Ver", id="frequency_map_button", className="mr-2", color="primary") ],
                            style={'textAlign': 'center'}
                        )
                        
                    ]),
                ),
            ]),




            #Card: Component plans
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de componentes",color="link",id="button_collapse_4"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_4",children=
                    dbc.CardBody(children=[
                        html.H5("Seleccionar atributos para mostar:"),
                        dcc.Dropdown(
                            id='dropdown_atrib_names',
                            options=session_data.get_data_features_names_dcc_dropdown_format(),
                            multi=True
                        ),
                        html.Div( 
                            [dbc.Checklist(
                                options=[{"label": "Seleccionar todos", "value": 1}],
                                value=[],
                                id="check_seleccionar_todos_mapas"),

                            dbc.Checklist(options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                       value=[],
                                       id="check_annotations_comp"),
                                        
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
                    html.H2(dbc.Button("Matriz U",color="link",id="button_collapse_5"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_5",children=
                    dbc.CardBody(children=[

                    html.H5("U-Matrix"),
                    html.H6("Returns the distance map of the weights.Each cell is the normalised sum of the distances betweena neuron and its neighbours. Note that this method usesthe euclidean distance"),
                    
                    html.Div(id='umatrix_figure_div', children=[''],style= pu.get_single_heatmap_css_style()
                    ),

                    html.Div([dbc.Button("Ver", id="umatrix_button", className="mr-2", color="primary"),
                            dbc.Checklist(options=[{"label": "Etiquetar Neuronas", "value": 1}],
                                       value=[],
                                       id="check_annotations_umax")
                            ],
                            style={'textAlign': 'center'}
                    )

                   
                    ])

                    ),
            ]),



            #Card: Guardar modelo
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Guardar modelo entrenado",color="link",id="button_collapse_6"),style={'textAlign': 'center'})
                ),
                dbc.Collapse(id="collapse_6",children=
                    dbc.CardBody(children=[
                  
                        html.Div(children=[
                            
                            html.H5("Nombre del fichero"),
                            dbc.Input(id='nombre_de_fichero_a_guardar_som',placeholder="Nombre del archivo", className="mb-3"),

                            dbc.Button("Guardar modelo", id="save_model_som", className="mr-2", color="primary"),
                            html.P('',id="check_correctly_saved_som")
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


def info_trained_params_som_table():

    info = session_data.get_som_model_info_dict()
    
    #Table
    table_header = [
         html.Thead(html.Tr([
                        html.Th("Tamaño Horizontal del Grid"),
                        html.Th("Tamaño Vertical del Grid"),
                        html.Th("Tasa Aprendizaje"),
                        html.Th("Función de Vecindad"),
                        html.Th("Función de Distancia"),
                        html.Th("Sigma Gaussiana"),
                        html.Th("Iteraciones"),
                        html.Th("Inicialización de Pesos"),
                        html.Th("Semilla")
        ]))
    ]

      
    if(info['check_semilla'] == 0):
        semilla = 'No'
    else:
        semilla = 'Sí: ' + str(info['seed']) 

    row_1 = html.Tr([html.Td( info['tam_eje_horizontal']),
                    html.Td( info['tam_eje_vertical']),
                     html.Td( info['learning_rate']) ,
                     html.Td( info['neigh_fun']),
                     html.Td( info['distance_fun']) ,
                     html.Td( info['sigma']) ,
                     html.Td( info['iteraciones'] ),
                     html.Td( info['inicialitacion_pesos']),
                     html.Td( semilla)

    ]) 

    table_body = [html.Tbody([row_1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children



##################################################################
#                       CALLBACKS
##################################################################

@app.callback(
    [Output(f"collapse_{i}", "is_open") for i in range(1, 7)],
    [Input(f"button_collapse_{i}", "n_clicks") for i in range(1, 7)],
    [State(f"collapse_{i}", "is_open") for i in range(1, 7)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4,n5,n6, is_open1, is_open2,is_open3,is_open4,is_open5,is_open6):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4, is_open5, is_open6
    elif button_id == "button_collapse_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4, is_open5, is_open6
    elif button_id == "button_collapse_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4, is_open5, is_open6
    elif button_id == "button_collapse_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4, is_open5, is_open6
    elif button_id == "button_collapse_5" and n5:
        return is_open1, is_open2, is_open3, is_open4, not is_open5, is_open6
    elif button_id == "button_collapse_6" and n6:
        return is_open1, is_open2, is_open3, is_open4, is_open5, not is_open6
    return False, False, False,False,False,False




#Habilitar boton ver_mapas_componentes_button
@app.callback(Output('ver_mapas_componentes_button','disabled'),
              Input('dropdown_atrib_names','value')
            )
def enable_ver_mapas_componentes_button(values):
    if ( values ):
        return False
    else:
        return True



#Estadisticas
@app.callback(Output('div_estadisticas_som', 'children'),
              Input('ver_estadisticas_som_button', 'n_clicks'),
              prevent_initial_call=True )
def ver_estadisticas_som(n_clicks):

    som = session_data.get_modelo()
    #data = session_data.get_data()
    data = session_data.get_data_std()


    qe,mqe = som.get_qe_and_mqe_errors(data)

    """tp : computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
    tp = som.topographic_error(data)
    

    #Table
    table_header = [
        html.Thead(html.Tr([html.Th("Magnitud"), html.Th("Valor")]))
    ]
    row0 = html.Tr([html.Td("Error de Cuantización"), html.Td(qe)])
    row1 = html.Tr([html.Td("Error de Cuantización Medio"), html.Td(mqe)])
    row2 = html.Tr([html.Td("Error Topográfico"), html.Td(tp)])
    table_body = [html.Tbody([row0,row1, row2])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children


#Etiquetar Mapa neuonas ganadoras
@app.callback(Output('winners_map', 'figure'),
              Input('check_annotations_winnersmap', 'value'),
              State('winners_map', 'figure'),
              State('ver', 'n_clicks'),
              prevent_initial_call=True )
def annotate_winners_map_som(check_annotations, fig,n_clicks):
    
    if(n_clicks is None):
        raise PreventUpdate
   
    if(check_annotations  ):
        fig_updated = pu.fig_add_annotations(fig)
    else:
        fig_updated = pu.fig_del_annotations(fig)

    return fig_updated


    


#Mapa neuonas ganadoras
@app.callback(Output('div_mapa_neuronas_ganadoras', 'children'),
              Input('ver', 'n_clicks'),
              State('check_annotations_winnersmap', 'value'),
              prevent_initial_call=True )
def update_som_fig(n_clicks, check_annotations):

    params = session_data.get_som_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']
    
 
    som = session_data.get_modelo()
    #dataset = session_data.get_dataset()
    #data = session_data.get_data()
    data = session_data.get_data_std()

    
    #targets = dataset[:,-1:]
    targets = session_data.get_targets_col()
    #targets_list = [t[0] for t in targets.tolist()]
    #TODO poner aqui .T en vez de to list
    targets_list =  targets.tolist()
    

    #'data and labels must have the same length.
    labels_map = som.labels_map(data, targets_list)
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    #labeled heatmap does not support nonetypes
    data_to_plot[:] = np.nan


    targets_freq = {}
    for t in targets_list:
        if (t in targets_freq):
            targets_freq[t] += 1
        else:
            targets_freq[t] = 1
    lista_targets_unicos = list(targets_freq.keys())
    #print('lista de targets unicos', lista_targets_unicos)

    
    if(session_data.get_discrete_data() ):
        #showing the class more represented in each neuron
        for position in labels_map.keys():
            label_fracs = [ labels_map[position][t] for t in lista_targets_unicos]
            max_value= max(label_fracs)
            winner_class_index = label_fracs.index(max_value)
            data_to_plot[position[0]][position[1]] = lista_targets_unicos[winner_class_index]
    else: #continuos data: mean of the mapped values in each neuron
        
        for position in labels_map.keys():
         
            #fractions
            label_fracs = [ labels_map[position][t] for t in lista_targets_unicos]
            #print('label_fracs', label_fracs)
            mean_div = sum(label_fracs)
            mean = sum([a*b for a,b in zip(lista_targets_unicos,label_fracs)])
            mean = mean/ mean_div
            data_to_plot[position[0]][position[1]] = mean
        

    fig = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations)
    children = pu.get_fig_div_with_info(fig,'winners_map', 'Mapa de neuronas ganadoras',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None)
    print('\nVISUALIZACION:renderfinalizado\n')

    return children


    

    

#Etiquetar freq map
@app.callback(Output('frequency_map', 'figure'),
              Input('check_annotations_freq', 'value'),
              State('frequency_map', 'figure'),
              State('frequency_map_button', 'n_clicks'),
              prevent_initial_call=True )
def annotate_freq_map_som(check_annotations, fig,n_clicks):
    
    if(n_clicks is None):
        raise PreventUpdate

    layout = fig['layout']
    data = fig['data']

    if(check_annotations  ): #fig already ploted
        trace = data[0]
        data_to_plot = trace['z'] 
        #To replace None values with NaN values
        data_to_plot_1 = np.array(data_to_plot, dtype=int)
        annotations = pu.make_annotations(data_to_plot_1, colorscale = DEFAULT_HEATMAP_COLORSCALE, reversescale= False)
        layout['annotations'] = annotations
    else:   
        layout['annotations'] = []

    fig_updated = dict(data=data, layout=layout)
    return fig_updated


#Actualizar mapas de frecuencias
@app.callback(Output('div_frequency_map','children'),
              Input('frequency_map_button','n_clicks'),
              State('check_annotations_freq', 'value'),
              prevent_initial_call=True 
              )
def update_mapa_frecuencias_fig(click, check_annotations):

    som = session_data.get_modelo() 
    #model_data = session_data.get_data()
    model_data = session_data.get_data_std()

    
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    tam_eje_vertical = params['tam_eje_vertical']

    frequencies = som.activation_response(model_data)
    frequencies_list = frequencies.tolist()
    
    figure = pu.create_heatmap_figure(frequencies_list,tam_eje_horizontal,tam_eje_vertical,check_annotations)

    children = pu.get_fig_div_with_info(figure,'frequency_map','Mapa de frecuencias',tam_eje_horizontal, tam_eje_vertical)

    return children
  


#Actualizar mapas de componentes
@app.callback(Output('component_plans_figures_div','children'),
              Input('ver_mapas_componentes_button','n_clicks'),
              State('dropdown_atrib_names','value'),
              State('check_annotations_comp', 'value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_fig(click,names,check_annotations):

    som = session_data.get_modelo()
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    tam_eje_vertical = params['tam_eje_vertical'] 

    nombres_atributos = session_data.get_only_features_names()
    lista_de_indices = []

    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    
    pesos = som.get_weights()
    traces = []
   
       
    for i in lista_de_indices:

        figure = pu.create_heatmap_figure(pesos[:,:,i].tolist() ,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = nombres_atributos[i])
        id ='graph-{}'.format(i)
        traces.append(html.Div(children= dcc.Graph(id=id,figure=figure)) )

    return traces
  



# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names','value'),
    Input("check_seleccionar_todos_mapas", "value"),
    prevent_initial_call=True
    )
def on_form_change(check):

    if(check):
        atribs = session_data.get_only_features_names()
        return atribs
    else:
        return []


      
#U-matrix
@app.callback(Output('umatrix_figure_div','children'),
              Input('umatrix_button','n_clicks'),
              Input('check_annotations_umax', 'value'),
              prevent_initial_call=True 
              )
def update_umatrix(n_clicks,check_annotations):

    if(n_clicks is None):
        raise PreventUpdate

    som = session_data.get_modelo()
    umatrix = som.distance_map()
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    tam_eje_vertical = params['tam_eje_vertical'] 
    figure = pu.create_heatmap_figure(umatrix.tolist() ,tam_eje_horizontal,tam_eje_vertical, check_annotations, title ='Matriz U',
                                         colorscale = UMATRIX_HEATMAP_COLORSCALE,  reversescale=True)
    return  html.Div(children= dcc.Graph(id='graph_u_matrix',figure=figure))





#Save file name
@app.callback(Output('nombre_de_fichero_a_guardar_som', 'valid'),
              Output('nombre_de_fichero_a_guardar_som', 'invalid'),
              Input('nombre_de_fichero_a_guardar_som', 'value'),
              prevent_initial_call=True
              )
def check_savesommodel_name(value):
    
    if not normpath(value) or search(r'[^A-Za-z0-9_\-]',value):
        return False,True
    else:
        return True,False




#Save SOM model
@app.callback(Output('check_correctly_saved_som', 'children'),
              Input('save_model_som', 'n_clicks'),
              State('nombre_de_fichero_a_guardar_som', 'value'),
              State('nombre_de_fichero_a_guardar_som', 'valid'),
              prevent_initial_call=True )
def save_som_model(n_clicks,name,isvalid):

    if(not isvalid):
        return ''

    data = []

    params = session_data.get_som_model_info_dict()
    columns_dtypes = session_data.get_colums_dtypes()

    data.append('som')
    data.append(columns_dtypes)
    data.append(params)
    data.append(session_data.get_modelo())

    filename =   name +  '_som.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Modelo guardado correctamente. Nombre del fichero: ' + filename