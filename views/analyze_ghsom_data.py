import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
import dash
import  views.elements as elements
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from math import ceil
import numpy as np
from collections import Counter
from datetime import datetime
import numpy.ma as ma

import plotly.graph_objects as go

from  views.session_data import session_data
from  config.config import DIR_SAVED_MODELS,UMATRIX_HEATMAP_COLORSCALE
import pickle

from  os.path import normpath 
from re import search 

import networkx as nx
import views.plot_utils as pu
from logging import raiseExceptions




##################################################################
#                       AUX LAYOUT FUNCTIONS
##################################################################

#Card Statistics
def get_statistics_card_ghsom():

    return  dbc.CardBody(children=[ 

                        html.Div(id = 'grafo_ghsom_estadisticas',children = '',
                                style=pu.get_css_style_inline_flex()
                        ),
                        html.Div( id='div_estadisticas_ghsom',children = '', style={'textAlign': 'center'}) 
            ])
            
            


#Card winners map
def get_winnersmap_card_ghsom():
    return  dbc.CardBody(children=[ 


                        dbc.Alert(
                        [
                            html.H4("Target not selected yet!", className="alert-heading"),
                            html.P(
                                "Please select a target below to print winners map. "
                            )
                        ],
                        color='danger',
                        id='alert_target_not_selected_ghsom',
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
                        id='output_alert_too_categorical_targets_ghsom',
                        is_open=False

                        ),


                        dcc.Dropdown(id='dropdown_target_selection_ghsom',
                           options=session_data.get_targets_options_dcc_dropdown_format() ,
                           multi=False,
                           value = session_data.get_target_name()
                        ),
                        html.Br(),  


                        dbc.Collapse(
                            id='collapse_winnersmap_ghsom',
                            is_open=False,
                            children=[

                                html.Div(id = 'grafo_ghsom_winners',children = '',
                                        style=pu.get_css_style_inline_flex()
                                ),
        
                                html.Div(   id = 'winners_map_ghsom',children = '',
                                            style= pu.get_single_heatmap_css_style()
                                ),
        
                                html.Div(
                                    dbc.Checklist(  options=[{"label": "Label Neurons", "value": 1}],
                                                    value=[],
                                                    id="check_annotations_winmap_ghsom"),
                                    style={'textAlign': 'center'}
                                ),

                                dbc.Collapse(
                                        id='collapse_logscale_winners_ghsom',
                                        is_open=session_data.is_preselected_target_numerical(),
                                        children=[
                                            dbc.FormGroup(
                                                [
                                                    dbc.RadioItems(
                                                        options=[
                                                            {"label": "Linear Scale", "value": 0},
                                                            {"label": "Logarithmic Scale", "value": 1},
                                                        ],
                                                        value=0,
                                                        id="radioscale_winners_ghsom",
                                                        inline=True,
                                                    ),
                                                ],
                                                style={'textAlign': 'center'}
                                            ),
                                        ]
                                )

                        ])

                    

            ])
        

#Card Freq Map
def get_freqmap_card_ghsom():
    return  dbc.CardBody(children=[ 

                        html.Div(children = [
                                html.Div(id = 'grafo_ghsom_freq',children = '',
                                        style={'textAlign': 'center'}
                                ),

                                html.Div(id = 'div_freq_map_ghsom',children=None
                                        #,style={'margin': '0 auto','width': '100%', 'display': 'flex','align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap', 'flex-direction': 'column ' } 
                                ),
                            ],style = pu.get_css_style_inline_flex()
                        ),


                        html.Div(children = [ 
                            dbc.FormGroup(
                                [
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Linear Scale", "value": 0},
                                            {"label": "Logarithmic Scale", "value": 1},
                                        ],
                                        value=0,
                                        id="radioscale_freq_ghsom",
                                        inline=True,
                                    ),
                                ],
                                style={'textAlign': 'center'}
                            )
                        ]),

                        html.Div(style=pu.get_css_style_inline_flex(),
                            children = [
                                html.H6( dbc.Badge( 'Minimum hits to plot a neuron   ' ,  pill=True, color="light", className="mr-1")   ),
                                html.H6( dbc.Badge( '0',  pill=True, color="warning", className="mr-1",id ='badge_min_hits_slider_ghsom')   ),
                            ]
                        ),            
                        dcc.Slider(id='min_hits_slider_ghsom', min=0,max=0,value=0,step =1 ),
            ])
          

#Card Component plans
def get_componentplans_card_ghsom():
      
    return  dbc.CardBody(children=[
                            html.H5("Seleccionar atributos para mostar:"),
                            dcc.Dropdown(
                                id='dropdown_atrib_names_ghsom',
                                options=session_data.get_data_features_names_dcc_dropdown_format(),
                                multi=True
                            ),
                            html.Div( 
                                [dbc.Checklist(
                                    options=[{"label": "Check All", "value": 1}],
                                    value=[],
                                    id="check_seleccionar_todos_mapas_ghsom")],
                                style={'textAlign': 'center'}
                            ),

                            
                            dbc.Alert(
                                    [
                                        html.H4("Feature(s) not selected yet!", className="alert-heading"),
                                        html.P(
                                            "Please select at least one feature below to plot his Component Plan. "
                                        )
                                    ],
                                    color='danger',
                                    id='alert_cplans_not_selected_ghsom',
                                    is_open=True

                            ),


                            dbc.Collapse(
                                id='collapse_cplan_ghsom',
                                is_open=False,
                                children=[

                                    #html.Div(id = 'grafo_ghsom_cplans',children = '',
                                    #    style=pu.get_css_style_inline_flex()
                                    #),

                                    html.Div(id='component_plans_figures_ghsom_div', children=[''],
                                            style= pu.get_single_heatmap_css_style()
                                    ),

                                    html.Div(dbc.Checklist(  options=[{"label": "Label Neurons", "value": 1}],
                                                            value=[],
                                                            id="check_annotations_comp_ghsom"),
                                                style={'textAlign': 'center'}
                                    ),

                                    dbc.FormGroup(
                                        [
                                            dbc.RadioItems(
                                                options=[
                                                    {"label": "Linear Scale", "value": 0},
                                                    {"label": "Logarithmic Scale", "value": 1},
                                                ],
                                                value=0,
                                                id="radioscale_cplans_ghsom",
                                                inline=True,
                                            ),
                                        ],
                                        style={'textAlign': 'center'}
                                    ),


                            ])

            ])
                    
# Card freq + cplans
def get_freq_and_cplans_cards_ghsom():
    children = []
    children.append(get_freqmap_card_ghsom())
    children.append(html.Hr())
    children.append(get_componentplans_card_ghsom())
    return html.Div(children)


#Card UMatrix
def get_umatrix_card_ghsom():
       
    return dbc.CardBody(children=[
            html.Div(id = 'grafo_ghsom_umatrix',children = '',
                    style=pu.get_css_style_inline_flex()
            ),
            
            html.Div(id = 'umatrix_div_fig_ghsom',children = '',
                    style= pu.get_single_heatmap_css_style()
            ),
            
            html.Div(dbc.Checklist(     options=[{"label": "Label Neurons", "value": 1}],
                                        value=[],
                                        id="check_annotations_um_ghsom"),
                    style={'textAlign': 'center'}
            ),

            dbc.FormGroup(
                [
                    dbc.RadioItems(
                        options=[
                            {"label": "Linear Scale", "value": 0},
                            {"label": "Logarithmic Scale", "value": 1},
                        ],
                        value=0,
                        id="radioscale_umatrix_ghsom",
                        inline=True,
                    ),
                ],
                style={'textAlign': 'center'}
            ),
        
        ])



#Card Guardar modelo
def get_savemodel_card_ghsom():

    return  dbc.CardBody(children=[
        
                html.Div(children=[

                    html.H5("Nombre del fichero"),
                    dbc.Input(id='nombre_de_fichero_a_guardar_ghsom',placeholder="Nombre del archivo", className="mb-3"),
                    dbc.Button("Guardar modelo", id="save_model_ghsom", className="mr-2", color="primary"),
                    html.P('',id="check_correctly_saved_ghsom")
                    ],
                    style={'textAlign': 'center'}
                ),
        ])
         






#############################################################
#	                       LAYOUT	                        #
#############################################################

def analyze_ghsom_data():

    # Body
    body =  html.Div(children=[
        html.H4('Data Analysis',className="card-title"  ),

        html.H6('Train Parameters',className="card-title"  ),
        html.Div(id = 'info_table_ghsom',children=info_trained_params_ghsom_table(),style={'textAlign': 'center'} ),



        dbc.Tabs(
                id='tabs_ghsom',
                active_tab='',
                style =pu.get_css_style_inline_flex(),
                children=[
                    dbc.Tab(get_select_splitted_option_card(),label = 'Select Splitted Dataset Part',tab_id='splitted_part',disabled= (not session_data.data_splitted )),
                    dbc.Tab(get_statistics_card_ghsom() ,label = 'Statistics',tab_id='statistics_card'),
                    dbc.Tab( get_winnersmap_card_ghsom() ,label = 'Winners Map',tab_id='winners_card'),
                    #dbc.Tab( get_freqmap_card_ghsom() ,label = 'Freq. Map',tab_id='freq_card'),
                    #dbc.Tab( get_componentplans_card_ghsom() ,label = 'Component Plans',tab_id='componentplans_card'),
                    dbc.Tab( get_freq_and_cplans_cards_ghsom(), label=' Freq. Map + Component Plans',tab_id='freq_and_cplans_ghsom'),
                    dbc.Tab( get_umatrix_card_ghsom() ,label = 'U Matrix',tab_id='umatrix_card'),
                    dbc.Tab( get_savemodel_card_ghsom() ,label = 'Save Model',tab_id='save_model_card'),
                    
                ]
        ),

        
    ])




    layout = html.Div(children=[

        elements.navigation_bar,
        body,
    ])

    return layout








##################################################################
#                       AUX FUNCTIONS
##################################################################


def info_trained_params_ghsom_table():

    info = session_data.get_ghsom_model_info_dict()

    
    #Table
    table_header = [
         html.Thead(html.Tr([
                        html.Th("Tau 1"),
                        html.Th("Tau 2"),
                        html.Th("Learning Rate"),
                        html.Th("Decadency"),
                        html.Th("Gaussian Sigma"),
                        html.Th("Epochs per Iteration"),
                        html.Th("Max. Iterations"),
                        html.Th("Dissimilarity Function"),
                        html.Th("Seed")
        ]))
    ]



    if(info['check_semilla'] == 0):
        semilla = 'No'
    else:
        semilla = 'Yes: ' + str(info['seed']) 
   
    row_1 = html.Tr([
                    html.Td( info['tau_1']) ,
                    html.Td( info['tau_2']) ,
                    html.Td( info['learning_rate']),
                    html.Td( info['decadency']) ,
                    html.Td( info['sigma']) ,
                    html.Td( info['epocas_gsom']) ,
                    html.Td( info['max_iter_gsom']  ),
                    html.Td( info['fun_disimilitud'] ),
                    html.Td( semilla )

    ]) 

    table_body = [html.Tbody([row_1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

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
def get_ghsom_fig(data, target_list):

    zero_unit = session_data.get_modelo()
    grafo = nx.Graph()
    g = zero_unit.child_map.get_structure_graph(grafo,data, target_list,level=0)
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
        hovertemplate= 'Neuron Parent Coordinates: <b>%{text}</b><br>'+
                        'Level: %{y}<br>' 
                        +"<extra></extra>"
        ,
        marker=dict(
            #color=['blue',], #set color equal to a variable
            color = 'slategray',
            size=14)
    )



    data1=[edge_trace, node_trace]


    layout = go.Layout(
            title="Complete GHSOM Structure",
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
        Aux. fun for ver_umatrix_ghsom_fig callbacks to optimize the calc. of umatrix
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



#Card select dataset part

def get_select_splitted_option_card():

    return     dbc.CardBody(   
                        children=[
                            dbc.Label("Select dataset portion"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Train Data", "value": 1},
                                    {"label": "Test Data", "value": 2},
                                    {"label": "Train + Test Data", "value": 3},
                                ],
                                value=2,
                                id="dataset_portion_radio_analyze_ghsom",
                            )
                        ]
                )






##################################################################
#                       CALLBACKS
##################################################################



#Toggle winners map if selected target
@app.callback(
    Output('alert_target_not_selected_ghsom', 'is_open'),
    Output('collapse_winnersmap_ghsom','is_open'),
    Output('dropdown_target_selection_ghsom', 'value'),
    Input('info_table_ghsom', 'children'), #udpate on load
    Input('dropdown_target_selection_ghsom', 'value'),
)
def toggle_winners_som(info_table,target_value):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    preselected_target = session_data.get_target_name()

    if( trigger_id == 'info_table_gsom' ): #init call
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
        return False, True,dash.no_update



#Toggle log scale option in winners map
@app.callback(
            Output('collapse_logscale_winners_ghsom', 'is_open'),
            Output('radioscale_winners_ghsom','radioscale_winners_som'),
            Input('dropdown_target_selection_ghsom', 'value'),
            State('dataset_portion_radio_analyze_ghsom','value'),
            #prevent_initial_call=True 
)  
def toggle_select_logscale_ghsom(target_value, option):

    if(target_value is None or not target_value):
        return False,0

    target_type,_ = session_data.get_selected_target_type( option)

    if(target_type is None or target_type == 'string'):
        return False,0

    else:
        return True,dash.no_update



#load ghsom graph
@app.callback(
    Output('grafo_ghsom_estadisticas','children'),
    Output('grafo_ghsom_winners','children'),
    #Output('grafo_ghsom_cplans','children'),
    Output('grafo_ghsom_umatrix','children'),
    Output('grafo_ghsom_freq','children'),
    #Input('tabs_ghsom', 'children'), #udpate on load
    Input("tabs_ghsom", "active_tab"),
    Input('dataset_portion_radio_analyze_ghsom', 'value'),
    Input('dropdown_target_selection_ghsom', 'value'),
    State('div_freq_map_ghsom', 'children'), 

    prevent_initial_call=True  
)
def load_graph_ghsom(tabs, option,target_selection,div_f):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if(trigger_id == 'tabs_ghsom' and div_f is not None ):
        raise PreventUpdate()
    else:

        print('\tLoading GHSOM Graph...', flush=True)
        data = session_data.get_data(option)
        if(session_data.get_target_name() is None):
            target_list = None
        else:
            target_list = session_data.get_targets_list(option)

        
        fig=  get_ghsom_fig(data, target_list)

        div_0 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_0')
        div_1 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_1')
        div_2 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_2')
        div_3 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_3')
        #div_4 = get_ghsom_graph_div(fig,'dcc_ghsom_graph_4')
        print('\tLoading Complete.', flush=True)

        #return div_0, div_1, div_2, div_3, div_2
        return div_0, div_1, div_3, div_2

        




#Estadisticas del ghsom  del punto seleccionado del grafo
@app.callback(Output('div_estadisticas_ghsom','children'),
              Output('dcc_ghsom_graph_0','figure'),
              Input('dcc_ghsom_graph_0','clickData'),
              State('dcc_ghsom_graph_0','figure'),
              State('dataset_portion_radio_analyze_ghsom', 'value'),
              prevent_initial_call=True 
              )
def view_stats_map_by_selected_point(clickdata,figure,option):

    if clickdata is  None:
        raise PreventUpdate

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


    #Estadisticas del gsom seleccionado

    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    gsom = nodes_dict[(cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)]
    data= session_data.get_data(option)
    gsom.replace_parent_dataset(data)
    qe, mqe = gsom.get_map_qe_and_mqe()
    params = session_data.get_ghsom_model_info_dict()
    fun_disimilitud = params['fun_disimilitud']
    


    #Table
    table_header = [
        html.Thead(html.Tr([html.Th("Magnitude"), html.Th("Value")]))
    ]
    if(fun_disimilitud == 'qe'):
        row0 = html.Tr([html.Td("MAPA: Sumatorio de  Errores de Cuantización(neuronas)"), html.Td(qe)])
        row1 = html.Tr([html.Td("MAPA: Promedio de  Errores de Cuantización(neuronas)"), html.Td(mqe)])
    else:
        row0 = html.Tr([html.Td("MAPA: Sumatorio de  Errores de Cuantización Medios(neuronas)"), html.Td(qe)])
        row1 = html.Tr([html.Td("MAPA: Promedio de  Errores de Cuantización Medios(neuronas)"), html.Td(mqe)])

    table_body = [html.Tbody([row0,row1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children,figure








#Winners map del punto seleccionado del grafo
@app.callback(Output('winners_map_ghsom','children'),
              Output('dcc_ghsom_graph_1','figure'),
              Output('output_alert_too_categorical_targets_ghsom','is_open'),
              Input('dcc_ghsom_graph_1','clickData'),
              Input('check_annotations_winmap_ghsom','value'),
              Input('radioscale_winners_ghsom','value'),
              State('dcc_ghsom_graph_1','figure'),
              State('dataset_portion_radio_analyze_ghsom','value'),
              prevent_initial_call=True 
              )
def view_winner_map_by_selected_point(clickdata, check_annotations,logscale, figure, data_portion_option):


    if clickdata is  None:
        raise PreventUpdate

    output_alert_too_categorical_targets_ghsom = False
    log_scale = False


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



    target_type,unique_targets = session_data.get_selected_target_type(data_portion_option)
    values=None 
    text = None


    if(target_type == 'numerical' ): #numerical data: mean of the mapped values in each neuron
        
        data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
        #labeled heatmap does not support nonetypes
        data_to_plot[:] = np.nan
        log_scale = logscale


        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in neurons_mapped_targets):
                        data_to_plot[i][j]  = np.mean(neurons_mapped_targets[(i,j)])
                else:
                    data_to_plot[i][j] = np.nan


    elif(target_type == 'string'):

        data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
        #labeled heatmap does not support nonetypes
        data_to_plot[:] = np.nan
        text = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
        #labeled heatmap does not support nonetypes
        text[:] = np.nan


        values = np.linspace(0, 1, len(unique_targets), endpoint=False).tolist()
        targets_codification = dict(zip(unique_targets, values))

        if(len(unique_targets) >= 270):
            output_alert_too_categorical_targets_ghsom = True


        #showing the class more represented in each neuron
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in neurons_mapped_targets):
                        c = Counter(neurons_mapped_targets[(i,j)])
                        #data_to_plot[i][j] = c.most_common(1)[0][0]
                        max_target= c.most_common(1)[0][0]
                        data_to_plot[i][j] =  targets_codification[max_target]
                        text[i][j] =max_target

                else:
                    data_to_plot[i][j] = np.nan

                
    else: #error
        raiseExceptions('Unexpected error')
        data_to_plot = np.empty([1 ,1],dtype= np.bool_)
        #labeled heatmap does not support nonetypes
        data_to_plot[:] = np.nan



    fig,table_legend = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations,
                                                text = text, discrete_values_range= values, unique_targets = unique_targets,
                                                title=None ,log_scale=log_scale)

    if(table_legend is not None):
        children = pu.get_fig_div_with_info(fig,'winnersmap_fig_ghsom', 'Winners Target per Neuron Map',tam_eje_horizontal, tam_eje_vertical,level,neurona_padre_string,
                                            table_legend =  table_legend)
    else:
        children = pu.get_fig_div_with_info(fig,'winnersmap_fig_ghsom', 'Winners Target per Neuron Map',tam_eje_horizontal, tam_eje_vertical,level,neurona_padre_string)


    print('\nGHSOM Winners Map Render finished\n')
    return children,figure, output_alert_too_categorical_targets_ghsom




#update_selected_min_hit_rate_badge_ghsom
@app.callback(  Output('badge_min_hits_slider_ghsom','children'),
                Input('min_hits_slider_ghsom','value'),
                prevent_initial_call=True 
)
def update_selected_min_hit_rate_badge_ghsom(value):
    return int(value)




#Frequency map
@app.callback(  Output('div_freq_map_ghsom','children'),
                #Output('dcc_ghsom_graph_4','figure'),
                Output('dcc_ghsom_graph_2','figure'),
                Output('min_hits_slider_ghsom','max'),
                Output('min_hits_slider_ghsom','marks'),
                Output('min_hits_slider_ghsom','value'),
                #Input('dcc_ghsom_graph_4','clickData'),
                Input('dcc_ghsom_graph_2','clickData'),
                Input('radioscale_freq_ghsom','value'),
                Input('min_hits_slider_ghsom','value'),
                #State('dcc_ghsom_graph_4','figure'),
                State('dcc_ghsom_graph_2','figure'),
                State('dataset_portion_radio_analyze_ghsom','value'),
                prevent_initial_call=True 
)
def update_freq_map_ghsom(clickdata, logscale, slider_value, figure, data_portion_option ):


    if clickdata is  None:
        raise PreventUpdate

    slider_update = dash.no_update
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


       
    #Mapa Freq  del gsom seleccionado en el grafo
    nodes_dict = session_data.get_ghsom_nodes_by_coord_dict()
    coords_nodes_dict = (cord_vertical_punto_clickeado,cord_horizontal_punto_clickeado)
    gsom = nodes_dict[coords_nodes_dict]
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()
    pre_calc_freq = session_data.get_calculated_freq_map()

    if( pre_calc_freq is None or (pre_calc_freq is not None and 
            (pre_calc_freq[1] != data_portion_option) or
            (pre_calc_freq[2] != coords_nodes_dict )
            ) ): #recalculate freq map
        
        g = session_data.get_ghsom_structure_graph()
        neurons_mapped_targets = g.nodes[gsom]['neurons_mapped_targets']
        data_to_plot = np.zeros([tam_eje_vertical ,tam_eje_horizontal],dtype=int)
    
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in neurons_mapped_targets):
                    c = len(neurons_mapped_targets[(i,j)])
                    data_to_plot[i][j] = c
                else:
                    data_to_plot[i][j] = 0


        session_data.set_calculated_freq_map(data_to_plot,data_portion_option,
                         coords_nodes_dict )

        slider_update = 0

    else:#load last calculated map
        data_to_plot,_,_ = pre_calc_freq

    
    max_freq = np.nanmax(data_to_plot)
    if(max_freq > 0):
        marks={
            0: '0 hits',
            int(max_freq): '{} hits'.format(int(max_freq))
        }
    else:
        marks = dash.no_update

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(slider_value != 0 and trigger_id == 'min_hits_slider_ghsom'):#filter minimum hit rate per neuron
        #frequencies = np.where(frequencies< slider_value,np.nan,frequencies)
        data_to_plot = ma.masked_less(data_to_plot, slider_value)
        session_data.set_freq_hitrate_mask(ma.getmask(data_to_plot))
    else:
        session_data.set_freq_hitrate_mask(None)


    fig,_ = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,True, title = None, log_scale = logscale)
    children = pu.get_fig_div_with_info(fig,'freq_map_ghsom', 'Frequency Map',tam_eje_horizontal, tam_eje_vertical)

    return children,figure,  max_freq,marks, slider_update




#Seleccionar al menos un atrib para poder plotear los component plans
@app.callback(
                Output('alert_cplans_not_selected_ghsom', 'is_open'),
                Output('collapse_cplan_ghsom', 'is_open'),
                Input('dropdown_atrib_names_ghsom', 'value'),
                prevent_initial_call=True
)
def show_graph_cplans(names):

    if(names is None or len(names)== 0):
        return True, False
    else:
        return False, True



# Checklist seleccionar todos mapas de componentes
@app.callback(
    Output('dropdown_atrib_names_ghsom','value'),
    Input("check_seleccionar_todos_mapas_ghsom", "value"),
    prevent_initial_call=True
    )
def on_form_change(check):

    if(check):
        atribs = session_data.get_features_names()
        return atribs
    else:
        return []





#Actualizar mapas de componentes
@app.callback(  Output('component_plans_figures_ghsom_div','children'),
                #Output('dcc_ghsom_graph_2','figure'),
                Input('div_freq_map_ghsom','children'),
                Input('check_annotations_comp_ghsom','value'),
                Input('radioscale_cplans_ghsom','value'),
                Input('min_hits_slider_ghsom','value'),
                Input('dropdown_atrib_names_ghsom','value'),
                State('dcc_ghsom_graph_2','figure'),
                State('dcc_ghsom_graph_2','clickData'),
                prevent_initial_call=True 
)
def update_mapa_componentes_ghsom_fig(input_freq_map,check_annotations,log_scale,slider_value,
                                        names, fig_grafo, clickdata ):


    if(clickdata is None or names is None or len(names) == 0):
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

    nombres_atributos = session_data.get_features_names()

    lista_de_indices = []
    for n in names:
        lista_de_indices.append(nombres_atributos.index(n) )
    


    traces = []
    

    for k in lista_de_indices:
        data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                data_to_plot[i][j] = weights_map[(i,j)][k]
        

        if(slider_value != 0 and session_data.get_freq_hitrate_mask() is not None):
            data_to_plot = ma.masked_array(data_to_plot, mask=session_data.get_freq_hitrate_mask() ).filled(np.nan)

        figure,_ =  pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations, 
                                                title = nombres_atributos[k],log_scale = log_scale)
        id ='graph-{}'.format(k)
        traces.append(
            html.Div(children= dcc.Graph(id=id,figure=figure)
            ) 
        )


    #return traces, fig_grafo
    return traces





#Ver UMatrix GHSOM
@app.callback(Output('umatrix_div_fig_ghsom','children'),
              Output('dcc_ghsom_graph_3','figure'),
              Input('dcc_ghsom_graph_3','clickData'),
              Input('check_annotations_um_ghsom','value'),
              Input('radioscale_umatrix_ghsom','value'),
              State('dcc_ghsom_graph_3','figure'),
              prevent_initial_call=True 
              )
def ver_umatrix_ghsom_fig(clickdata,check_annotations,log_scale, fig_grafo):


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
    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
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

 
    fig, _ = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = None,
                                    colorscale = UMATRIX_HEATMAP_COLORSCALE,  reversescale=True,  log_scale = log_scale)
    children= pu.get_fig_div_with_info(fig,'umatrix_fig_ghsom', 'U-Matrix',tam_eje_horizontal, tam_eje_vertical)
    print('\n GHSOM UMatrix: Plotling complete! \n')

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
    columns_dtypes = session_data.get_features_dtypes()
    
    data.append('ghsom')
    data.append(columns_dtypes)
    data.append(params)
    data.append(session_data.get_modelo())

    filename =   name +  '_ghsom.pickle'
    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Model saved! Filename: ' + filename




