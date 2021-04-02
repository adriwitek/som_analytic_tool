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
import pickle
from  os.path import normpath 
from re import search 
import views.plot_utils as pu


fig = go.Figure()




def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Análisis de los datos',className="card-title"  ),
        html.Hr(),
        html.Div(children=[ 

            #Card Mapa neurona winners
            #TODO div con la fig
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Mapa de neuronas ganadoras",color="link",id="button_collapse_1"))
                ),
                dbc.Collapse(id="collapse_1",children=
                    dbc.CardBody(children=[ 
                        html.Div([  dcc.Graph(id='winners_map',figure=fig)],
                                    id='div_mapa_neuronas_ganadoras',
                                    style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                        ),
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
            ]),



            #Card: Guardar modelo
            dbc.Card([
                dbc.CardHeader(
                    html.H2(dbc.Button("Guardar modelo entrenado",color="link",id="button_collapse_5"))
                ),
                dbc.Collapse(id="collapse_5",children=
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
#                       CALLBACKS
##################################################################

@app.callback(
    [Output(f"collapse_{i}", "is_open") for i in range(1, 6)],
    [Input(f"button_collapse_{i}", "n_clicks") for i in range(1, 6)],
    [State(f"collapse_{i}", "is_open") for i in range(1, 6)],
    prevent_initial_call=True)
def toggle_accordion(n1, n2,n3,n4,n5, is_open1, is_open2,is_open3,is_open4,is_open5):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "button_collapse_1" and n1:
        return not is_open1, is_open2, is_open3,is_open4, is_open5
    elif button_id == "button_collapse_2" and n2:
        return is_open1, not is_open2, is_open3,is_open4, is_open5
    elif button_id == "button_collapse_3" and n3:
        return is_open1, is_open2, not is_open3,is_open4, is_open5
    elif button_id == "button_collapse_4" and n4:
        return is_open1, is_open2, is_open3, not is_open4, is_open5
    elif button_id == "button_collapse_5" and n5:
        return is_open1, is_open2, is_open3, is_open4, not is_open5
    return False, False, False,False,False


#Habilitar boton ver_mapas_componentes_button
@app.callback(Output('ver_mapas_componentes_button','disabled'),
              Input('dropdown_atrib_names','value')
            )
def enable_ver_mapas_componentes_button(values):
    if ( values ):
        return False
    else:
        return True


#Etiquetar Mapa neuonas ganadoras
@app.callback(Output('div_mapa_neuronas_ganadoras', 'children'),
              Input('check_annotations_winnersmap', 'value'),
              State('winners_map', 'figure'),
              State('ver', 'n_clicks'),

              prevent_initial_call=True )
def annotate_winners_map_som(check_annotations, fig,n_clicks):
    
    if(n_clicks is None):
        raise PreventUpdate

    params = session_data.get_som_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']

    layout = {}
    layout['title'] = 'Mapa de neuronas ganadoras'
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }

    data = fig['data']


    if(check_annotations  ): #fig already ploted
        trace = data[0]
        data_to_plot = trace['z'] 
        #To replace None values with NaN values
        data_to_plot_1 = np.array(data_to_plot, dtype=np.float64)

        annotations = pu.make_annotations(data_to_plot_1, colorscale = 'Jet', reversescale= False)
        layout['annotations'] = annotations
        
   

    

    fig_updated = dict(data=data, layout=layout)

    children = [dcc.Graph(id='winners_map',figure=fig_updated)]
    return children
        
    


#Mapa neuonas ganadoras
@app.callback(Output('winners_map', 'figure'),
              Input('ver', 'n_clicks'),
              State('check_annotations_winnersmap', 'value'),
              prevent_initial_call=True )
def update_som_fig(n_clicks, check_annotations):

    print('\nVISUALIZACION clicked\n')


    params = session_data.get_som_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']
    
    #TODO : cambiar esto por guardado bien del dataset

    som = session_data.get_modelo()
    dataset = session_data.get_dataset()
    data = session_data.get_data()
    targets = dataset[:,-1:]
    

    #print('targets',[t for t in targets])
    targets_list = [t[0] for t in targets.tolist()]
    #print('targetssss',targets_list)
    labels_map = som.labels_map(data, targets_list)
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
    #labeled heatmap does not support nonetypes
    print('debug 1')
    data_to_plot[:] = np.nan

    if(session_data.get_discrete_data() ):
        #showing the class more represented in each neuron
        for position in labels_map.keys():
            label_fracs = [ labels_map[position][t] for t in targets_list]
            max_value= max(label_fracs)
            winner_class_index = label_fracs.index(max_value)
            data_to_plot[position[0]][position[1]] = targets_list[winner_class_index]
    else: #continuos data: mean of the mapped values in each neuron
        for position in labels_map.keys():
            #fractions
            label_fracs = [ labels_map[position][t] for t in targets_list]
            data_to_plot[position[0]][position[1]] = np.mean(label_fracs)

    

    print('debug 2')


    '''


    x_ticks = np.linspace(0, tam_eje_vertical,tam_eje_vertical, dtype= int,endpoint=False).tolist()
    y_ticks = np.linspace(0, tam_eje_horizontal,tam_eje_horizontal,dtype= int, endpoint=False ).tolist()

    ######################################
    # ANNOTATED HEATMAPD LENTO
    #colorscale=[[np.nan, 'rgb(255,255,255)']]
    #fig = ff.create_annotated_heatmap(
    '''
    '''
    fig = custom_heatmap(
        #x= x_ticks,
        #y= y_ticks,
        z=data_to_plot,
        zmin=np.nanmin(data_to_plot),
        zmax=np.nanmax(data_to_plot),
        #xgap=5,
        #ygap=5,
        colorscale='Viridis',
        #colorscale=colorscale,
        #font_colors=font_colors,
        
        showscale=True #leyenda de colores
        )
    fig.update_layout(title_text='Clases ganadoras por neurona')
    fig['layout'].update(plot_bgcolor='white')
    '''

    
    #########################################################
    # ANNOTATED HEATMAPD RAPIDOOO
    
    #TODO
    #type= heatmap para mas precision
    #heatmapgl
    #trace = dict(type='heatmapgl', z=data_to_plot, colorscale = 'Jet')
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
    layout['title'] = 'Mapa de neuronas ganadoras'
    layout['xaxis']  ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    layout['yaxis'] ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
    #layout['width'] = 700
    #layout['height']= 700

    fig = dict(data=data, layout=layout)

    #condition_Nones = not(val is None)
    #condition_nans= not(np.isnan(val))




    if(check_annotations):
        print('Empezando a anotar')
        annotations = pu.make_annotations(data_to_plot, colorscale = 'Jet', reversescale= False)
        print('Fin de la anotacion')
        layout['annotations'] = annotations
    

    print('\nVISUALIZACION:renderfinalizado\n')

    return fig









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


    params = session_data.get_som_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical'] 
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    nombres_columnas = datos_entrenamiento['columns_names']
    nombres_atributos = nombres_columnas[0:len(nombres_columnas)-1]
    lista_de_indices = []
    print('Las  dimensiones del mapa entrenado son:',tam_eje_vertical,tam_eje_horizontal)


    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    

    pesos = som.get_weights()

    traces = []





    xaxis_dict ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    yaxis_dict  ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
       


    for i in lista_de_indices:
        
        #figure= go.Figure(layout= {"height": 300,'width' : 300, 'title': nombres_atributos[i], 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict},
        figure= go.Figure(layout= { 'title': nombres_atributos[i], 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict},
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




    


    
#Actualizar mapas de frecuencias
@app.callback(Output('frequency_map','figure'),
              Input('frequency_map_button','n_clicks'),
              prevent_initial_call=True 
              )
def update_mapa_frecuencias_fig(click):

    som = session_data.get_modelo() 
    som =  session_data.get_modelo()
    dataset = session_data.get_dataset()
    data = dataset[:,:-1]
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 

    xaxis_dict ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    yaxis_dict  ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
       


    #frequencies is a np matrix
    frequencies = som.activation_response(data)
    figure= go.Figure(layout= {'title': 'Mapa de frecuencias absolutas', 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict},
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
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 

    xaxis_dict ={'tickformat': ',d', 'range': [-0.5,(tam_eje_horizontal-1)+0.5] , 'constrain' : "domain"}
    yaxis_dict  ={'tickformat': ',d', 'scaleanchor': 'x','scaleratio': 1 }
       

    figure= go.Figure(layout= {'title': 'Matriz U', 'xaxis': xaxis_dict, 'yaxis' : yaxis_dict},
                          data=go.Heatmap(z=umatrix.tolist(),showscale= True)                              
    ) 


    print('render finalizado')
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




#Save GSOM model
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

    data.append('som')
    data.append(params)
    data.append(session_data.get_modelo())

    '''
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M")
    filename = 'gsom_model_' + dt_string + '.pickle'
    '''
    filename =   name +  '_som.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Modelo guardado correctamente. Nombre del fichero: ' + filename