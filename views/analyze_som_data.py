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
from logging import raiseExceptions

from plotly.colors import validate_colors

'''
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
'''

import matplotlib as mpl
import matplotlib.cm as cm

#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################

#ESTADISTICAS
def get_estadisticas_som_card():
    return  dbc.CardBody(children=[ 
                        html.Div( id='div_estadisticas_som',children = '', style={'textAlign': 'center'}),
                        html.Div([
                            dbc.Button("Plot", id="ver_estadisticas_som_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ])
            



#Mapa neurona winners
def get_mapaneuronasganadoras_som_card():

    return  dbc.CardBody(children=[ 

                    dbc.Alert(
                        [
                            html.H4("Target not selected yet !", className="alert-heading"),
                            html.P(
                                "Please select a target below to print winners map. "
                            )
                        ],
                        color='danger',
                        id='alert_target_not_selected_som',
                        is_open=True

                    ),

                    dbc.Alert(
                        [
                            html.H4("Too many different categorial targets !", className="alert-heading"),
                            html.P(
                                "Since there are more than 269(max. diferenciable discrete colors) unique targets, color representation will be ambiguous for some of them. "
                            )
                        ],
                        color='danger',
                        id='output_alert_too_categorical_targets',
                        is_open=False

                    ),


                    dcc.Dropdown(id='dropdown_target_selection_som',
                           options=session_data.get_targets_options_dcc_dropdown_format() ,
                           multi=False,
                           value = session_data.get_target_name()
                    ),
                    html.Br(),    

                    dbc.Collapse(
                        id='collapse_winnersmap_som',
                        is_open=False,
                        children=[

                            html.Div( id='div_mapa_neuronas_ganadoras',children = '', style= pu.get_single_heatmap_css_style()),
                                html.Div([
                                    dbc.Checklist(  options=[{"label": "Label Neurons", "value": 1}],
                                                    value=[],
                                                    id="check_annotations_winnersmap"),
                                    dbc.Button("Plot", id="ver", className="mr-2", color="primary")],
                                    style={'textAlign': 'center'}
                                )
                            ])
            

            ])

#Mapa frecuencias
def get_mapafrecuencias_som_card():

    return  dbc.CardBody(children=[
                html.Div( id='div_frequency_map',children = '',style= pu.get_single_heatmap_css_style()),
                html.Div([ 
                    dbc.Checklist(options=[{"label": "Label Neurons", "value": 1}],
                                    value=[],
                                    id="check_annotations_freq"),
                                
                    dbc.Button("Plot", id="frequency_map_button", className="mr-2", color="primary") ],
                    style={'textAlign': 'center'}
                )
                
            ])
            




#Card: Component plans
def get_componentplans_som_card():

    return  dbc.CardBody(children=[
                        html.H5("Select Features"),
                        dcc.Dropdown(
                            id='dropdown_atrib_names',
                            options=session_data.get_data_features_names_dcc_dropdown_format(),
                            multi=True
                        ),
                        html.Div( 
                            [dbc.Checklist(
                                options=[{"label": "Select all", "value": 1}],
                                value=[],
                                id="check_seleccionar_todos_mapas"),

                            dbc.Checklist(options=[{"label": "Label Neurons", "value": 1}],
                                       value=[],
                                       id="check_annotations_comp"),
                                        
                            dbc.Button("Plot Selected Components Map", id="ver_mapas_componentes_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),
                        html.Div(id='component_plans_figures_div', children=[''],
                                style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}
                        )

                ])



#Card: U Matrix
def get_umatrix_som_card():

    return  dbc.CardBody(children=[

                    html.H5("U-Matrix"),
                    html.H6("Returns the distance map of the weights.Each cell is the normalised sum of the distances betweena neuron and its neighbours. Note that this method usesthe euclidean distance"),
                    
                    html.Div(id='umatrix_figure_div', children=[''],style= pu.get_single_heatmap_css_style()
                    ),

                    html.Div([dbc.Button("Plot", id="umatrix_button", className="mr-2", color="primary"),
                            dbc.Checklist(options=[{"label": "Label Neurons", "value": 1}],
                                       value=[],
                                       id="check_annotations_umax")
                            ],
                            style={'textAlign': 'center'}
                    )

                   
            ])

              


#Card: Guardar modelo
def get_savemodel_som_card():

    return dbc.CardBody(children=[
                  
                        html.Div(children=[
                            
                            html.H5("Filename"),
                            dbc.Input(id='nombre_de_fichero_a_guardar_som',placeholder="Filename", className="mb-3"),

                            dbc.Button("Save Model", id="save_model_som", className="mr-2", color="primary"),
                            html.P('',id="check_correctly_saved_som")
                            ],
                            style={'textAlign': 'center'}
                        ),
            ])
          

def get_select_splitted_option_card():

    return     dbc.CardBody(   id='get_select_splitted_option_card_som',
                        children=[
                            dbc.Label("Select dataset portion"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Train Data", "value": 1},
                                    {"label": "Test Data", "value": 2},
                                    {"label": "Train + Test Data", "value": 3},
                                ],
                                value=2,
                                id="dataset_portion_radio_analyze_som",
                            )
                        ]
                )




#############################################################
#	                       LAYOUT	                        #
#############################################################



def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Data Analysis \n',className="card-title"  ),

        html.H6('Train Parameters',className="card-title"  ),
        html.Div(id = 'info_table_som',children=info_trained_params_som_table(),style={'textAlign': 'center'} ),


     
     
      

        dbc.Tabs(
            id='tabs_som',
            active_tab='components_plans_som',
            style ={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'},
            children=[
                dbc.Tab(get_select_splitted_option_card(),label = 'Select Dataset Splitted Part',tab_id='splitted_part',disabled= (not session_data.data_splitted )),
                dbc.Tab(get_estadisticas_som_card(),label = 'Statistics',tab_id='statistics_som' ),
                dbc.Tab(get_mapaneuronasganadoras_som_card(),label = 'Winners Target Map',tab_id='winners_map_som'),
                dbc.Tab( get_mapafrecuencias_som_card() ,label = 'Freq',tab_id='freq_som'),
                dbc.Tab( get_componentplans_som_card(), label='Component Plans',tab_id='components_plans_som'),
                dbc.Tab(get_umatrix_som_card()  , label='U-Matrix',tab_id='umatrix_som'),
                dbc.Tab(get_savemodel_som_card() ,label = 'Save Model',tab_id='save_model_som'),
            ]
        ),

     
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
                        html.Th("Grid Horizontal Size"),
                        html.Th("Grid Vertical Size"),
                        html.Th("Learning Rate"),
                        html.Th("Neighborhood Function"),
                        html.Th("Distance Function"),
                        html.Th("Gaussian Sigma"),
                        html.Th("Max Iterations"),
                        html.Th("Weights Initialization"),
                        html.Th("Topology"),
                        html.Th("Seed")
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
                     html.Td( info['topology']),
                     html.Td( semilla)

    ]) 

    table_body = [html.Tbody([row_1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children








##################################################################
#                       CALLBACKS
##################################################################

'''

#Show select splitted target part if it es splitted
@app.callback(  Output('get_select_splitted_option_card_som', 'is_open'),
                Input('info_table_som', 'children'), #udpate on load
)
def toggle_show_dataset_splitted_option(info_table):


    dbc.CardBody(children=[
                html.Div( id='div_frequency_map',children = '',style= pu.get_single_heatmap_css_style()),
                html.Div([ 
                    dbc.Checklist(options=[{"label": "Label Neurons", "value": 1}],
                                    value=[],
                                    id="check_annotations_freq"),
                                
                    dbc.Button("Plot", id="frequency_map_button", className="mr-2", color="primary") ],
                    style={'textAlign': 'center'}
                )
                
            ])

    print('debugdsfa',flush=True)

    if(session_data.data_splitted):

        
        return True
    else:
        return False
'''

#Toggle winners map if selected target
@app.callback(
    Output('alert_target_not_selected_som', 'is_open'),
    Output('collapse_winnersmap_som','is_open'),
    Output('dropdown_target_selection_som', 'value'),
    Input('info_table_som', 'children'), #udpate on load
    Input('dropdown_target_selection_som', 'value'),
)
def toggle_winners_som(info_table,target_value):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    preselected_target = session_data.get_target_name()

    if( trigger_id == 'info_table_som' ): #init call
        if(preselected_target is None):
            return False,True, None

        else:
            return False,True, preselected_target

    elif(target_value is None or not target_value):
        session_data.set_target_name(None)
        #session_data.targets_col = []
        return True,False, None


    else:
        session_data.set_target_name(target_value)
        #df = pd.read_json(origina_df,orient='split')
        #df = df[target_value]
        #session_data.targets_col = df_target.to_numpy()
        
        return False, True,dash.no_update

    

#TODO BORRAR
'''
@app.callback(
                Output('dropdown_target_selection_som', 'options'),
                Input('info_table_som', 'children'), #udpate on load
                State('original_dataframe_storage','data')
)
def get_target_optins(info_table, origina_df):

    df = pd.read_json(origina_df,orient='split',nrows = 1)
    df = df[exclude = session_data.get_features_names()]
    columnas = df.columns.tolist()

    options = []  # must be a list of dicts per option

    for n in columnas:
        options.append({'label' : n, 'value': n})
    return options
'''




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
              State('dataset_portion_radio_analyze_som','value'),
              prevent_initial_call=True )
def ver_estadisticas_som(n_clicks,data_portion_option):

    som = session_data.get_modelo()
    #data = session_data.get_data_std()
    data = session_data.get_data(data_portion_option)

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
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))
    ]
    row0 = html.Tr([html.Td("Quantization Error"), html.Td(qe)])
    row1 = html.Tr([html.Td("Average Quantization Error"), html.Td(mqe)])
    row2 = html.Tr([html.Td("Topographic Error"), html.Td(tp)])
    table_body = [html.Tbody([row0,row1, row2])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children


'''
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

'''
    


#Mapa neuonas ganadoras
@app.callback(Output('div_mapa_neuronas_ganadoras', 'children'),
              Output('output_alert_too_categorical_targets', 'is_open'),
              Input('ver', 'n_clicks'),
              Input('check_annotations_winnersmap', 'value'),
              State('dataset_portion_radio_analyze_som','value'),
              prevent_initial_call=True )
def update_som_fig(n_clicks, check_annotations, data_portion_option):


    output_alert_too_categorical_targets = False
    params = session_data.get_som_model_info_dict()
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']
 
    som = session_data.get_modelo()
    data = session_data.get_data(data_portion_option)

    targets_list = session_data.get_targets_list(data_portion_option)
    #'data and labels must have the same length.
    labels_map = som.labels_map(data, targets_list)
    

    if(params['topology']== 'rectangular'):    #RECTANGULAR TOPOLOGY    

        target_type,unique_targets = session_data.get_selected_target_type(data_portion_option)
        values=None 
        text = None


        if(target_type == 'numerical' ): #numerical data: mean of the mapped values in each neuron

            data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
            #labeled heatmap does not support nonetypes
            data_to_plot[:] = np.nan
            #text = None

            for position in labels_map.keys():
            
                label_fracs = labels_map[position]
                #denom = sum(label_fracs.values())
                numerador = 0.0
                denom = 0.0
                for k,it in label_fracs.items():
                    numerador = numerador + k*it
                    denom = denom + it

                mean = numerador / denom
                data_to_plot[position[0]][position[1]] = mean


                '''
                #fractions
                label_fracs = [ labels_map[position][t] for t in lista_targets_unicos]
                #print('label_fracs', label_fracs)
                mean_div = sum(label_fracs)
                mean = sum([a*b for a,b in zip(lista_targets_unicos,label_fracs)])
                mean = mean/ mean_div
                data_to_plot[position[0]][position[1]] = mean
                '''

                '''
                elif(target_type == 'boolean'):
                
                    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype= np.intc)
                    #labeled heatmap does not support nonetypes
                    data_to_plot[:] = np.nan
                    text = None


                    for position in labels_map.keys():
                    
                        true_freqs = labels_map[position]['True']
                        false_freqs = labels_map[position]['False']

                        if(true_freqs >=false_freqs ):
                            data_to_plot[position[0]][position[1]] = 1
                        else:
                            data_to_plot[position[0]][position[1]] = 0
                '''


        elif(target_type == 'string'):

            data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
            #labeled heatmap does not support nonetypes
            data_to_plot[:] = np.nan
            text = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
            #labeled heatmap does not support nonetypes
            text[:] = np.nan
            values = np.linspace(0, 1, len(unique_targets), endpoint=False).tolist()
            targets_codification = dict(zip(unique_targets, values))
            #print('targets codificados',targets_codification)

            if(len(unique_targets) >= 270):
                output_alert_too_categorical_targets = True



            #showing the class more represented in each neuron
            for position in labels_map.keys():
                max_target = max(labels_map[position], key=labels_map[position].get)
                data_to_plot[position[0]][position[1]] = targets_codification[max_target]
                text[position[0]][position[1]] = max_target


        else: #error
            raiseExceptions('Unexpected error')
            data_to_plot = np.empty([1 ,1],dtype= np.bool_)
            #labeled heatmap does not support nonetypes
            data_to_plot[:] = np.nan
            #text = None



        fig,table_legend = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations,
                                         text = text, discrete_values_range= values, unique_targets = unique_targets)
        if(table_legend is not None):
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Target per Neuron Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                                neurona_padre=None,  table_legend =  table_legend)
        else:
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Target per Neuron Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None)
        print('\n SOM Winning Neuron Map: Plotling complete! \n')



    else: #HEXAGONAL TOPOLOGY

    
        print('MODO HEXAGONAL....',flush=True)

       
        xx, yy = som.get_euclidean_coordinates()

        xx_list = []
        yy_list = []
        zz_list = []


        for position in labels_map.keys():
                print('Hit POSICION:',position)
            
                label_fracs = labels_map[position]
                #denom = sum(label_fracs.values())
                numerador = 0.0
                denom = 0.0
                mean = 0.0
                for k,it in label_fracs.items():
                    numerador = numerador + k*it
                    denom = denom + it

                mean = numerador / denom
              
                wx = xx[position]
                wy = yy[position]

                xx_list.append(wx) 
                yy_list.append(wy)
                zz_list.append(mean)


        #testing all cells
        #xx_list = np.nditer(xx)
        #yy_list = np.nditer(yy)

        shapes = []
        centers_x = []
        centers_y = []
        counts = []


        norm = mpl.colors.Normalize(vmin=min(zz_list), vmax= max(zz_list))
        cmap = cm.jet
        '''
        cmap(norm(5))

        colormap= pu.get_normalized_colormap(vmin=min(zz_list), vmax= max(zz_list))
        color = matplotlib.colors.rgb2hex(colormap.to_rgba(0.5))
        '''
        print('cmap de 0.55', cmap(0.5),flush=True)
    
        for i,j,z in zip(xx_list, yy_list,zz_list):
            rgb =  cmap(norm(z))[:3]
            color = mpl.colors.rgb2hex(rgb)
            shape = pu.create_hexagon(i,j,color   )
            shapes.append(shape)
            centers_x.append(i)
            centers_y.append(j)
            counts.append(z)


        #print('centers_x',centers_x)
        #print('centers_y',centers_y)
        #print('counts',counts)


        #define  text to be  displayed on hovering the mouse over the cells
        text = [f'x: {centers_x[k]}<br>y: {centers_y[k]}<br>Value: {counts[k]}' for k in range(len(centers_x))]
        pl_jet_color = pu.mpl_to_plotly(cm.jet, len(counts))


        trace = go.Scatter(
             x=list(centers_x), 
             y=list(centers_y), 
             mode='markers',
             marker=dict(size=0.5, 
                         color=counts, 
                         colorscale=pl_jet_color, 
                         showscale=True,
                         colorbar=dict(
                                    thickness=20,  
                                     ticklen=4
                                     )
                         ),             
           text=text, 
           hoverinfo='text'
        )    


        axis = dict(showgrid=False,
           showline=False,
           zeroline=False,
           ticklen=4 
           )

        layout = go.Layout(title='Hexbin plot',
                   width=530, height=550,
                   xaxis=axis,
                   yaxis=axis,
                   hovermode='closest',
                   shapes=shapes,
                   plot_bgcolor=None)


        #fig = go.FigureWidget(data=[trace], layout=layout)
        data=[trace]
        data.append({'type': 'scattergl',
                    'mode': 'text'
                })

        fig = dict(data=data, layout=layout)


        children = pu.get_fig_div_with_info(fig,'winners_map', 'PRUEBA HEXAGONAL MAPA',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None)
        print('\n TESTTTTTTTTTSOM Winning Neuron Map: Plotling complete! \n')




    return children, output_alert_too_categorical_targets



    

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
@app.callback(  Output('div_frequency_map','children'),
                Input('frequency_map_button','n_clicks'),
                State('check_annotations_freq', 'value'),
                State('dataset_portion_radio_analyze_som','value'),
                prevent_initial_call=True 
)
def update_mapa_frecuencias_fig(click, check_annotations, data_portion_option):

    som = session_data.get_modelo() 
   
    model_data =  session_data.get_data(data_portion_option)

    
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    tam_eje_vertical = params['tam_eje_vertical']

    frequencies = som.activation_response(model_data)
    frequencies_list = frequencies.tolist()
    
    figure = pu.create_heatmap_figure(frequencies_list,tam_eje_horizontal,tam_eje_vertical,check_annotations)

    children= pu.get_fig_div_with_info(figure,'frequency_map','Frequency Map',tam_eje_horizontal, tam_eje_vertical)

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

    nombres_atributos = session_data.get_features_names()
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
        atribs = session_data.get_features_names()
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
    columns_dtypes = session_data.get_features_dtypes()

    data.append('som')
    data.append(columns_dtypes)
    data.append(params)
    data.append(session_data.get_modelo())

    filename =   name +  '_som.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Model saved! Filename: ' + filename