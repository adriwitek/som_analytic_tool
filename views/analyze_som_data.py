from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Collapse import Collapse
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from flask.globals import session
from views.app import app
import dash
import  views.elements as elements
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil
import numpy as np
import numpy.ma as ma


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
import dash_table



#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################




def create_new_model_row(qe, h_size, v_size, lr, nf, df,gs,mi,wi,t,s, training_time):
    row = {}
    row['QE'] = qe
    row['Horizontal Size'] = h_size
    row['Vertical Size'] = v_size
    row['Learning Rate'] = lr
    row['Neighborhood Function'] = nf
    row['Distance Function'] = df
    row['Gaussian Sigma'] = gs
    row['Max Iterations'] = mi
    row['Weights Initialization'] = wi
    row['Topology'] = t
    row['Seed'] = s
    row['training_time'] = training_time


    return row


def create_multi_soms_table():
            
    columns = []
    columns.append({'id': 'QE'         ,             'name': 'QE' })
    columns.append({'id': 'Training Time'         ,  'name': 'Training Time' })
    columns.append({'id': 'Horizontal Size'         , 'name': 'Hor. Size' })
    columns.append({'id': 'Vertical Size'           , 'name': 'Ver. Size' })
    columns.append({'id': "Learning Rate"           , 'name': "Learning Rate" })
    columns.append({'id': "Neighborhood Function"   , 'name': "Neighborhood Function" })
    columns.append({'id': "Distance Function"       , 'name': "Distance Function"})
    columns.append({'id': "Gaussian Sigma"          , 'name': "Gaussian Sigma"})
    columns.append({'id': "Max Iterations"          , 'name': "Max Iterations"})
    columns.append({'id': "Weights Initialization"  , 'name': "Weights Initialization"})
    columns.append({'id': "Topology"                , 'name': "Topology"})
    columns.append({'id': "Seed"                    , 'name': "Seed"})
    


    data = []

    for som_params in session_data.get_som_models_info_dict() :


        v_size    = som_params['tam_eje_vertical'] 
        h_size =   som_params['tam_eje_horizontal'] 
        lr       =   som_params['learning_rate']  
        nf    =   som_params['neigh_fun'] 
        df    =   som_params['distance_fun'] 
        gs    =   som_params['sigma'] 
        mi    =   som_params['iteraciones'] 
        wi    =   som_params['inicialitacion_pesos'] 
        t    =   som_params['topology'] 
        check_semilla    =   som_params['check_semilla'] 
        semilla    =   som_params['seed'] 
        training_time    =   som_params['training_time'] 

        if(check_semilla):
           seed = 'Yes:' + str(semilla)
        else:
            seed = 'No'

        if(som_params['qe'] is None):
            qe = 'qe not avaible'
        else:
            qe = som_params['qe'] 

        row = create_new_model_row(qe, h_size, v_size, lr, nf, df,gs,mi,wi,t,seed, training_time)
        data.append(row)


    table =  dash_table.DataTable(	id = 'table_analyze_multi_som',
                                    data = data,
                                    columns = columns,
                                    row_selectable='single',
                                    selected_rows = [0],
                                    row_deletable=False,
                                    editable=False,
                                    sort_action='native',
                                    style_cell={      'textAlign': 'center',
                                                      'textOverflow': 'ellipsis',
                                                      'overflowX': 'auto'
                                    },
                                    #style_as_list_view=True,
                                    style_header={
                                            'backgroundColor': 'rgb(255, 255, 53)',
                                            'fontWeight': 'bold'
                                    },

                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        },

                                        {
                                            'if': {'row_index': [0]},
                                            'backgroundColor': '#D2F3FF',
                                            #'fontWeight': 'bold',
                                            #'color': 'white',
                                        },
                                    ],
       
    )


    tablediv = html.Div(children=table, style = {"overflow": "scroll"})
                                        
    return tablediv







#############################################################
#	                       LAYOUT	                        #
#############################################################

def get_model_selection_card():


    return    dbc.Card(children=[
            
                dbc.CardBody(
                    children = [

                        dbc.Button("Show Model Info/ Change Model Selection",id="show_model_selection_button",
                                className="mb-6",color="success",block=True),
                        dbc.Collapse(   children = create_multi_soms_table(),
                                        id= 'collapse_model_selection',
                                        is_open=False,
                        ),
                        html.Div(id = 'hidden_div_analyze_som'),
                    ],
                )
            ],
            #color="secondary",
        )


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

                                    dbc.Collapse(
                                        id='collapse_logscale_winners_som',
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
                                                        id="radioscale_winners_som",
                                                        inline=True,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),

                                   
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

                    dbc.FormGroup(
                        [
                            dbc.RadioItems(
                                options=[
                                    {"label": "Linear Scale", "value": 0},
                                    {"label": "Logarithmic Scale", "value": 1},
                                ],
                                value=0,
                                id="radioscale_freq_som",
                                inline=True,
                            ),
                        ]
                    ),
                                
                    html.Div(style=pu.get_css_style_inline_flex(),
                        children = [
                            html.H6( dbc.Badge( 'Minimum hits to plot a neuron   ' ,  pill=True, color="light", className="mr-1")   ),
                            html.H6( dbc.Badge( '0',  pill=True, color="warning", className="mr-1",id ='badge_min_hits_slider_som')   ),
                        ]
                    ),            
                    dcc.Slider(id='min_hits_slider_som', min=0,max=0,value=0,step =1 ),
                    
                    dbc.Button("Plot Freq. Map", id="frequency_map_button", className="mr-2", color="primary") ],
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

                            dbc.FormGroup(
                                [
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Linear Scale", "value": 0},
                                            {"label": "Logarithmic Scale", "value": 1},
                                        ],
                                        value=0,
                                        id="radioscale_cplans_som",
                                        inline=True,
                                    ),
                                ]
                            ),

                            dbc.Alert(
                                children =[
                                    html.H4("Model have changed!", className="alert-heading"),
                                    html.P(
                                        "Replot Frequency Map or set Minimum hit rate to 0, to be able to plot Component plans. "
                                    )
                                ],
        
                                id="alert_fade_cplans_som",
                                color='danger',
                                dismissable=True,
                                is_open=False,
                            ),
                                        
                            dbc.Button("Plot Selected Components Map", id="ver_mapas_componentes_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),
                        html.Div(id='component_plans_figures_div', children=[''],
                                style=pu.get_css_style_inline_flex()
                        )

                ])

# Card freq + cplans
def get_freq_and_cplans_cards():
    children = []
    children.append(get_mapafrecuencias_som_card())
    children.append(html.Hr())
    children.append(get_componentplans_som_card())
    return html.Div(children)




#Card: U Matrix
def get_umatrix_som_card():

    return  dbc.CardBody(children=[

                    html.H5("U-Matrix"),
                    html.H6("Returns the distance map of the weights.Each cell is the normalised sum of the distances betweena neuron and its neighbours. Note that this method usesthe euclidean distance"),
                    
                    html.Div(id='umatrix_figure_div', children=[''],style= pu.get_single_heatmap_css_style()
                    ),

                    html.Div(
                        [
                            dbc.Checklist(options=[{"label": "Label Neurons", "value": 1}],
                                       value=[],
                                       id="check_annotations_umax"),

                            dbc.FormGroup(
                                [
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Linear Scale", "value": 0},
                                            {"label": "Logarithmic Scale", "value": 1},
                                        ],
                                        value=0,
                                        id="radioscale_umatrix_som",
                                        inline=True,
                                    ),
                                ]
                            ),
                            dbc.Button("Plot", id="umatrix_button", className="mr-2", color="primary"),       

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







def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Data Analysis \n',className="card-title"  ),



        #TODO BORRAR ESTO Y LA FUNCION DE LA TABLA DE AHORA!!!!!!
        #html.H6('Train Parameters',className="card-title"  ),
        #html.Div(id = 'info_table_som',children=info_trained_params_som_table(),style=pu.get_css_style_center() ),

        #Model Selection Card
        html.Div(id ='div_model_selection_card',children = get_model_selection_card()),
     
     
    
        dbc.Tabs(
            id='tabs_som',
            active_tab='winners_map_som',
            style =pu.get_css_style_inline_flex(),
            children=[
                dbc.Tab(get_select_splitted_option_card(),label = 'Select Splitted Dataset Part',tab_id='splitted_part',disabled= (not session_data.data_splitted )),
                dbc.Tab(get_estadisticas_som_card(),label = 'Statistics',tab_id='statistics_som' ),
                dbc.Tab(get_mapaneuronasganadoras_som_card(),label = 'Winners Target Map',tab_id='winners_map_som'),
                #dbc.Tab( get_mapafrecuencias_som_card() ,label = 'Freq',tab_id='freq_som'),
                #dbc.Tab( get_componentplans_som_card(), label='Component Plans',tab_id='components_plans_som'),
                dbc.Tab( get_freq_and_cplans_cards(), label=' Freq. Map + Component Plans',tab_id='freq_and_cplans_som'),
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
        semilla = 'Yes: ' + str(info['seed']) 

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


@app.callback(  Output('collapse_model_selection','is_open'),
                Input('show_model_selection_button', 'n_clicks'),
                State('collapse_model_selection','is_open'),
                prevent_initial_call=True 
)

def open_collapse_model_selection(n_clicks,is_open):
    return not is_open



@app.callback(  Output('hidden_div_analyze_som','children'),
                Output( 'table_analyze_multi_som', 'style_data_conditional'),
                Input( 'table_analyze_multi_som', 'selected_rows'),
                State( 'table_analyze_multi_som', 'style_data_conditional'),


)
def update_selected_row_session_data(selected_rows, style_data_conditional):
    session_data.set_selected_model_index(selected_rows[0])
    session_data.reset_calculated_freq_map()

    style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            },
                            {
                                'if': { 'row_index': selected_rows[0] },
                                'background_color': '#D2F3FF'
                            },

                            ]

    return '', style_data_conditional




#Toggle winners map if selected target
@app.callback(
    Output('alert_target_not_selected_som', 'is_open'),
    Output('collapse_winnersmap_som','is_open'),
    Output('dropdown_target_selection_som', 'value'),
    Input('div_model_selection_card', 'children'), #udpate on load
    Input('dropdown_target_selection_som', 'value'),
)
def toggle_winners_som(info_table,target_value):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    preselected_target = session_data.get_target_name()

    if( trigger_id == 'div_model_selection_card' ): #init call
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

    




#Habilitar boton ver_mapas_componentes_button
@app.callback(Output('ver_mapas_componentes_button','disabled'),
              Input('dropdown_atrib_names','value'),
              Input('min_hits_slider_som','value'),
            )
def enable_ver_mapas_componentes_button(values,slider_value):
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
    data = session_data.get_data(data_portion_option)

    qe,mqe = som.get_qe_and_mqe_errors(data)

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
#deprecated
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

#Toggle log scale option in winners map
@app.callback(
            Output('collapse_logscale_winners_som', 'is_open'),
            Output('radioscale_winners_som','radioscale_winners_som'),
            Input('dropdown_target_selection_som', 'value'),
            State('dataset_portion_radio_analyze_som','value'),
            #prevent_initial_call=True 
)  
def toggle_select_logscale(target_value, option):

    if(target_value is None or not target_value):
        return False,0

    target_type,_ = session_data.get_selected_target_type( option)

    if(target_type is None or target_type == 'string'):
        return False,0

    else:
        return True,dash.no_update



#Mapa neuronas ganadoras
@app.callback(Output('div_mapa_neuronas_ganadoras', 'children'),
              Output('output_alert_too_categorical_targets', 'is_open'),
              Input('ver', 'n_clicks'),
              Input('check_annotations_winnersmap', 'value'),
              Input('radioscale_winners_som','value'),
              State('dataset_portion_radio_analyze_som','value'),
              prevent_initial_call=True )
def update_som_fig(n_clicks, check_annotations ,logscale, data_portion_option):


    output_alert_too_categorical_targets = False
    params = session_data.get_som_model_info_dict()
    log_scale = False
    tam_eje_vertical = params['tam_eje_vertical']
    tam_eje_horizontal = params['tam_eje_horizontal']
 
    som = session_data.get_modelo()
    data = session_data.get_data(data_portion_option)

    targets_list = session_data.get_targets_list(data_portion_option)
    #'data and labels must have the same length.
    labels_map = som.labels_map(data, targets_list)
    
    target_type,unique_targets = session_data.get_selected_target_type(data_portion_option)
    discrete_values_range=None 
    text = None


    if(params['topology']== 'rectangular'):    #RECTANGULAR TOPOLOGY    

    
        if(target_type == 'numerical' ): #numerical data: mean of the mapped values in each neuron

            data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
            #labeled heatmap does not support nonetypes
            data_to_plot[:] = np.nan
            log_scale = logscale

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



        elif(target_type == 'string'):

            data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
            #labeled heatmap does not support nonetypes
            data_to_plot[:] = np.nan
            text = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=object)
            #labeled heatmap does not support nonetypes
            text[:] = np.nan
            discrete_values_range = np.linspace(0, 1, len(unique_targets), endpoint=False).tolist()
            targets_codification = dict(zip(unique_targets, discrete_values_range))
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
            data_to_plot[:] = np.nan
            



        fig,table_legend = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations,
                                                    text = text, discrete_values_range= discrete_values_range, unique_targets = unique_targets,
                                                    log_scale = log_scale)
        if(table_legend is not None):
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Target per Neuron Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                                neurona_padre=None,  table_legend =  table_legend)
        else:
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Target per Neuron Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                                neurona_padre=None)
        print('\n SOM Winning Neuron Map: Plotling complete! \n')



    else: ###########  HEXAGONAL TOPOLOGY

        xx, yy = som.get_euclidean_coordinates()
        xx_list = []
        yy_list = []
        zz_list = []
        text_list = None

        if(target_type == 'numerical' ): #numerical data: mean of the mapped values in each neuron
            log_scale = logscale

            for position in labels_map.keys():

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


        elif(target_type == 'string'):

            text_list = []
    
            #values = np.linspace(0, 1, len(unique_targets), endpoint=False).tolist()
            discrete_values_range = range(0,len(unique_targets))
            targets_codification = dict(zip(unique_targets, discrete_values_range))

            if(len(unique_targets) >= 270):
                output_alert_too_categorical_targets = True

            #showing the class more represented in each neuron
            for position in labels_map.keys():
                max_target = max(labels_map[position], key=labels_map[position].get)

                wx = xx[position]
                wy = yy[position]
                xx_list.append(wx) 
                yy_list.append(wy)
                zz_list.append(targets_codification[max_target])
                text_list.append(max_target)



        else: #error
            raiseExceptions('Unexpected error')
               


        fig,table_legend = pu.create_hexagonal_figure(xx_list,yy_list,zz_list, hovertext= True,log_scale = log_scale,
                                                     check_annotations =check_annotations, text_list = text_list,
                                                     discrete_values_range= discrete_values_range,
                                                     unique_targets = unique_targets)
        if(table_legend is not None):
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                                neurona_padre=None,  table_legend =  table_legend)
        else:
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                                neurona_padre=None)
        
        #children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None)
        print('\n SOM(hexagonal) Winning Neuron Map: Plotling complete! \n')




    return children, output_alert_too_categorical_targets



    
'''
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
    params = session_data.get_som_model_info_dict()

    
    if(params['topology']== 'rectangular'):    #RECTANGULAR TOPOLOGY   

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
    
    else:

        return dash.no_update

'''


#update_selected_min_hit_rate_badge_som
@app.callback(  Output('badge_min_hits_slider_som','children'),
                Input('min_hits_slider_som','value'),
                prevent_initial_call=True 
)
def update_selected_min_hit_rate_badge_som(value):
    return int(value)


#Actualizar mapas de frecuencias
@app.callback(  Output('div_frequency_map','children'),
                Output('min_hits_slider_som','max'),
                Output('min_hits_slider_som','marks'),
                Output('min_hits_slider_som','value'),
                Input('frequency_map_button','n_clicks'),
                Input('check_annotations_freq', 'value'),
                Input('radioscale_freq_som','value'),
                Input('min_hits_slider_som','value'),
                State('dataset_portion_radio_analyze_som','value'),
                prevent_initial_call=True 
)
def update_mapa_frecuencias_fig(click, check_annotations ,log_scale ,slider_value, data_portion_option):


    som = session_data.get_modelo() 
    model_data =  session_data.get_data(data_portion_option)

    
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    tam_eje_vertical = params['tam_eje_vertical']
    pre_calc_freq = session_data.get_calculated_freq_map()

    #ctx = dash.callback_context
    #trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if(    pre_calc_freq is None or 
            (pre_calc_freq is not None and pre_calc_freq[1] != data_portion_option) ): #recalculate freq map
        frequencies = som.activation_response(model_data)
        session_data.set_calculated_freq_map(frequencies,data_portion_option )
        slider_value = 0
        #frequencies_list = frequencies.tolist()

    else:#load last calculated map
        frequencies,_, _ = pre_calc_freq
  

    max_freq = np.nanmax(frequencies)
    if(max_freq > 0):
        marks={
            0: '0 hits',
            int(max_freq): '{} hits'.format(int(max_freq))
        }
    else:
        marks = dash.no_update

    if(slider_value != 0):#filter minimum hit rate per neuron
        #frequencies = np.where(frequencies< slider_value,np.nan,frequencies)
        frequencies = ma.masked_less(frequencies, slider_value)
        session_data.set_freq_hitrate_mask(ma.getmask(frequencies))
    else:
        session_data.set_freq_hitrate_mask(None)



    if(params['topology']== 'rectangular'):    #RECTANGULAR TOPOLOGY 
        figure,_ = pu.create_heatmap_figure(frequencies,tam_eje_horizontal,tam_eje_vertical,
                                            check_annotations,log_scale = log_scale)

    else:#Hexagonal topology
        xx, yy = som.get_euclidean_coordinates()
        xx_list = xx.ravel()
        yy_list = yy.ravel()

        if(slider_value != 0):
            #zz_list = frequencies.astype(float).filled(np.nan).ravel() #error in colorscale
            zz_list = frequencies.filled(-999).ravel()
            zz_list   = [i if i>0 else np.nan for i in zz_list]
        else:
            zz_list = frequencies.ravel()

        figure,_ = pu.create_hexagonal_figure(xx_list,yy_list, zz_list, hovertext= True,
                                             check_annotations =check_annotations,log_scale = log_scale )
        

    children= pu.get_fig_div_with_info(figure,'frequency_map','Frequency Map',tam_eje_horizontal, tam_eje_vertical)

    return children, max_freq,marks,slider_value
  


#Actualizar mapas de componentes
@app.callback(  Output('component_plans_figures_div','children'),
                Output('alert_fade_cplans_som','is_open'),
                Input('ver_mapas_componentes_button','n_clicks'),
                Input('min_hits_slider_som','value'),
                State('dropdown_atrib_names','value'),
                State('check_annotations_comp', 'value'),
                State('radioscale_cplans_som','value'),
                State( 'table_analyze_multi_som', 'selected_rows'),
                prevent_initial_call=True 
)
def update_mapa_componentes_fig(n_cliks,slider_value,names,check_annotations, log_scale,selected_rows):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if( (trigger_id == 'min_hits_slider_som' and n_cliks == 0)):
        raise PreventUpdate
    elif(trigger_id == 'min_hits_slider_som' and ( names is None or  len(names)==0)):
        return dash.no_update, dash.no_update
    elif( n_cliks != 0 and session_data.get_calculated_freq_map() is None and slider_value !=0 ):
        return dash.no_update, True

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


    if(params['topology']== 'rectangular'):    #RECTANGULAR TOPOLOGY   

    
        for i in lista_de_indices:
            #pesos[:,:,i].tolist()
            cplan = pesos[:,:,i]
            if(slider_value != 0 and session_data.get_freq_hitrate_mask() is not None):
                cplan = ma.masked_array(cplan, mask=session_data.get_freq_hitrate_mask() )

            figure,_ = pu.create_heatmap_figure(cplan ,tam_eje_horizontal,tam_eje_vertical,check_annotations,
                                                 title = nombres_atributos[i], log_scale = log_scale)
            id ='graph-{}'.format(i)
            traces.append(html.Div(children= dcc.Graph(id=id,figure=figure)) )

    else: #Hexagonal topology

        xx, yy = som.get_euclidean_coordinates()
        xx_list = xx.ravel()
        yy_list = yy.ravel()

        for i in lista_de_indices:
            cplan = pesos[:,:,i]

            if(slider_value != 0 and session_data.get_freq_hitrate_mask() is not None):
                cplan = ma.masked_array(cplan, mask=session_data.get_freq_hitrate_mask() )
                zz_list = cplan.filled(np.nan).ravel() 
            else:
                zz_list = cplan.ravel()
            figure,_ = pu.create_hexagonal_figure(xx_list,yy_list, zz_list, hovertext= True, title = nombres_atributos[i],
                                                 check_annotations= check_annotations, log_scale = log_scale)
            id ='graph-{}'.format(i)
            traces.append(html.Div(children= dcc.Graph(id=id,figure=figure)) )


    return traces, False
  



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
              Input('radioscale_umatrix_som','value'),
              prevent_initial_call=True 
              )
def update_umatrix(n_clicks,check_annotations, log_scale):

    if(n_clicks is None):
        raise PreventUpdate

    som = session_data.get_modelo()
    umatrix = som.distance_map()
    params = session_data.get_som_model_info_dict()
    tam_eje_horizontal = params['tam_eje_horizontal'] 
    tam_eje_vertical = params['tam_eje_vertical'] 

    params = session_data.get_som_model_info_dict()

    if(params['topology']== 'rectangular'):    #RECTANGULAR TOPOLOGY   
        figure,_ = pu.create_heatmap_figure(umatrix ,tam_eje_horizontal,tam_eje_vertical, check_annotations, title ='Matriz U',
                                             colorscale = UMATRIX_HEATMAP_COLORSCALE,  reversescale=True, log_scale = log_scale)
    else:#HEXAGONAL TOPOLOGY

        xx, yy = som.get_euclidean_coordinates()
        xx_list = xx.ravel()
        yy_list = yy.ravel()
        zz_list = umatrix.ravel()
        figure, _ =  pu.create_hexagonal_figure(xx_list,yy_list,zz_list, hovertext= True, colorscale = UMATRIX_HEATMAP_COLORSCALE,
                                             check_annotations = check_annotations, log_scale = log_scale)
      

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