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

import numpy as np
import numpy.ma as ma


from  views.session_data import session_data
from  config.config import *
#from  config.config import  DIR_SAVED_MODELS, UMATRIX_HEATMAP_COLORSCALE
import pickle
from  os.path import normpath 
from re import search 
import views.plot_utils as pu
from logging import raiseExceptions

import time
import base64
from pathlib import Path
import pandas as pd
import os
from pathlib import Path
import io
import csv
import dash_table



#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################




def create_new_model_row(qe, h_size, v_size, lr, nf, df,gs,mi,wi,t,s, training_time,tpe):
    row = {}
    row['MQE'] = qe
    row['tpe'] = tpe
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
    if(isinstance(training_time,float) ):
        row['Training Time'] = time.strftime("%H h %M m %S s", time.gmtime(training_time))
    else:
        row['Training Time'] =training_time

    return row


def create_multi_soms_table():
            
    columns = []
    columns.append({'id': 'MQE'         ,             'name': 'MQE' })
    columns.append({'id': 'tpe'         ,             'name': 'Topographic Error' })
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

        if(som_params['mqe'] is None):
            qe = 'Not Calculated'
        else:
            qe = som_params['mqe'] 

        if(som_params['tpe'] is None):
            tpe = 'Not Calculated'
        else:
            tpe = som_params['tpe'] 

        row = create_new_model_row(qe, h_size, v_size, lr, nf, df,gs,mi,wi,t,seed, training_time,tpe)
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
                        dbc.Collapse(   children =[ create_multi_soms_table(),
                                                    html.P('Replot Graps after select a new model',className="text-secondary",  style= pu.get_css_style_center() ),
                                        ] ,
                                        id= 'collapse_model_selection',
                                        is_open=False,
                        ),
                        html.Div(id = 'hidden_div_analyze_som'),
                    ],
                )
            ],
            #color="secondary",
        )


#Statistics
def get_estadisticas_som_card():
    return  dbc.CardBody(children=[ 
                        html.Div( id='div_estadisticas_som',children = '', style={'textAlign': 'center'}),
                        html.Div([
                            dbc.Button("Calculate", id="ver_estadisticas_som_button", className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        )
                    ])
            



#MAP: Winners Map
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

#Freq mat
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

                    #html.H5("U-Matrix"),                    
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
                        style=pu.get_css_style_center()
                    )

                   
            ])

#Card: Anomaly Detetection
def get_anomaly_detection_card():  

    return dbc.CardBody(children=[
                        html.Div(children=[
                            
                            html.H5("Anomaly Detection, tutoriall......."),
                            html.Br(),
                            html.H5('Upload File to Search for Anomalies in Data',className="card-title" , style=pu.get_css_style_center() ),
                            dcc.Upload( id='upload_anomaly_data', 
                                        children= pu.get_upload_data_component_text() ,
                                        style={'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'},
                                        # Allow multiple files to be uploaded
                                        multiple=False
                            ),
                            dbc.Collapse(   id = 'collapse_info_loaded_file_anomaly',
                                            children = '',
                                            is_open = False,
                            ),
                            dbc.Collapse(   id = 'collapse_error_loaded_file_anomaly',
                                            children = '',
                                            is_open = False,
                            ),
                            
                            html.Br(),

                            
                            #Normality percentage
                            html.Div(style=pu.get_css_style_inline_flex(),
                                    children = [
                                        html.H6(children='Minimum Normality percentage\t'),
                                        #html.H6( dbc.Badge( '1 %',  pill=True, color="warning", className="mr-1",id ='badge_info_normalitypercentage_slider')   )
                                        dcc.Input(id="normality_percentage", type="number", value=0.01,step=0.0000001,min=0, max = 1),
                                    ]
                            ),
                            html.P('Small values increasing from 0 recommended, for an optimum search  ',className="text-secondary" ),
                            html.P('0  means All Data will be Classified as Normal ',className="text-secondary" ),
                            html.P('1  means All Data will be Classified as Anomaly ',className="text-secondary" ),

                            html.Br(),
                            dbc.Collapse(   id = 'collapse_anomaly_result_table',
                                            children = '',
                                            is_open = False,
                                            style = pu.get_css_style_center()
                            ),

                           
                            
                            #Search for anomalies button
                            dcc.Loading(id='loading',
                                    type='dot',
                                    children=[
                                        dbc.Button("Search for Anomalies", id="anomaly_button",disabled = True,  className="mr-2", color="warning",),
                                    ]
                            ),
                            html.Br(),
                            html.Br(),
                            dbc.Collapse(   id = 'collapse_anomaly_result_table',
                                            children = pu.create_simple_table([], [], 'table_detected_anomalies'),
                                            is_open = False,
                            ),
                            html.Br(),
                            #Save to file anomlies menu
                            dbc.Collapse(   id = 'collapse_show_save_found_anomalies_to_file',
                                            is_open = False,
                                            children =[
                                                dbc.Button("Save Results Menu", id="save_anomalies_showmenu_button",disabled = True,  className="mr-2", color="info",),
                                            ],
                                            style= pu.get_css_style_center()
                            ),
                            #Filename and save button
                            dbc.Collapse(   id = 'collapse_filename_save_anomalies',
                                            is_open =False,
                                            children = [
                                                html.H5("Filename"),
                                                dbc.Input(id='filename_anomaly_rows_input',placeholder="Filename", className="mb-3"),
                                                dbc.Button("Save", id="filename_anomaly_rows_button", className="mr-2", color="success"),
                                                html.P('',id="correctly_saved_anomalies_rows")

                                            ],
                                            style= pu.get_css_style_center()


                            ),


        
                            ],
                            style=pu.get_css_style_center()
                        ),
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
                            style=pu.get_css_style_center()
                        ),
            ])
          

def get_select_splitted_option_card():
 
    return  dbc.CardBody(  id='get_select_splitted_option_card_som', 
                        children=[
                            html.Div(
                                children = [
                                    html.H5( dbc.Badge( 'Select Dataset Portion' ,  color="info", className="mr-1")   ),
                                    html.Br(),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Train Data", "value": 1},
                                            {"label": "Test Data", "value": 2},
                                            {"label": "Train + Test Data", "value": 3},
                                        ],
                                        value=2,
                                        id="dataset_portion_radio_analyze_som",
                                    ),
                                    html.Br(),
                                    html.P('Replot Graps after select a new option',className="text-secondary" ),
                                ],
                                style={'display': 'inline-block', 'text-align': 'left'},
                            ),
                            
                        ],
                        style = pu.get_css_style_center(),
                )







def analyze_som_data():

    # Body
    body =  html.Div(children=[
        html.H4('Data Analysis \n',className="card-title"  ),

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
                dbc.Tab(get_anomaly_detection_card() ,label = 'Anomaly Detection',tab_id='anomaly_detection_som'),
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
def update_selected_row_session_data(selected_rows, style_data_conditional, ):
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



#Stats
@app.callback(  Output('div_estadisticas_som', 'children'),
                Output( 'table_analyze_multi_som', 'data'),
                Input('ver_estadisticas_som_button', 'n_clicks'),
                State('dataset_portion_radio_analyze_som','value'),
                State( 'table_analyze_multi_som', 'selected_rows'),
                State( 'table_analyze_multi_som', 'data'),
                prevent_initial_call=True 
)
def ver_estadisticas_som(n_clicks,data_portion_option, selected_rows, table_data):

    som = session_data.get_modelo()
    data = session_data.get_data(data_portion_option)

    qe,mqe = som.get_qe_and_mqe_errors(data)
    tp = som.topographic_error(data)
    qe = round(qe, 4)
    mqe = round(mqe, 4)
    tp = round(tp,4) 
    #Table
    table_header = [
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))
    ]
    row0 = html.Tr([html.Td("Quantization Error(Total)"), html.Td(qe)])
    row1 = html.Tr([html.Td("Mean Quantization Error"), html.Td(mqe)])
    row2 = html.Tr([html.Td("Topographic Error"), html.Td(tp)])
    table_body = [html.Tbody([row0,row1, row2])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]


    update = False
    if(table_data[selected_rows[0]]['MQE'] == 'Not Calculated' ):
        table_data[selected_rows[0]]['MQE'] = mqe
        update = True

        #return children, table_data

    if(table_data[selected_rows[0]]['tpe'] == 'Not Calculated'):
        table_data[selected_rows[0]]['tpe'] = tp
        update = True


    if(update):
        return children, table_data
    else:
        return children, dash.no_update


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
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Target per Neuron',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                                neurona_padre=None,  table_legend =  table_legend)
        else:
            children = pu.get_fig_div_with_info(fig,'winners_map', 'Winners Target per Neuron',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
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
        figure,_ = pu.create_heatmap_figure(umatrix ,tam_eje_horizontal,tam_eje_vertical, check_annotations,
                                             colorscale = UMATRIX_HEATMAP_COLORSCALE,  reversescale=True, log_scale = log_scale)
    else:#HEXAGONAL TOPOLOGY

        xx, yy = som.get_euclidean_coordinates()
        xx_list = xx.ravel()
        yy_list = yy.ravel()
        zz_list = umatrix.ravel()
        figure, _ =  pu.create_hexagonal_figure(xx_list,yy_list,zz_list, hovertext= True, colorscale = UMATRIX_HEATMAP_COLORSCALE,
                                             check_annotations = check_annotations, log_scale = log_scale)
      
    
    return html.Div(pu.get_fig_div_with_info(figure,'graph_u_matrix','U-Matrix',tam_eje_horizontal, tam_eje_vertical)) 

    #return  html.Div(children= dcc.Graph(id='graph_u_matrix',figure=figure))





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





#TODO DELETE THIS
'''
##################ANOMALY DETECTION CALLBACKS
# Sync slider wit badge info min normality percentage
@app.callback(  Output("badge_info_normalitypercentage_slider", "children"),
                #Input("normality_percentage_slider", "value"),
                Input("normality_percentage", "value"),
                prevent_initial_call=True
                )
def sync_slider_input_datasetpercentage(v):

    if(v is None):
        return '0 %'
    else:
        return (str(v) + ' %')
   
'''

#Carga la info del dataset en el home
@app.callback( 
                Output('collapse_error_loaded_file_anomaly', 'is_open'),
                Output('collapse_error_loaded_file_anomaly', 'children'),
                Output('collapse_info_loaded_file_anomaly', 'is_open'),
                Output('collapse_info_loaded_file_anomaly', 'children'),
                Output('upload_anomaly_data', 'children'),#Just for training spinner animation
                Input('upload_anomaly_data', 'contents'),
                State('upload_anomaly_data', 'filename'),
                prevent_initial_call=True
)
def update_output( contents, filename):
    '''Loads anomaly test dataset

    '''
    if contents is not None:
        try:
            if 'csv' in filename:
                content_split = contents.split(',')
                if(len(content_split) <2):
                    return True,'An error occurred processing the file' , False, '', pu.get_upload_data_component_text()
                content_type, content_string = content_split
                decoded = base64.b64decode(content_string)
                try:
                    # Assume that the user uploaded a CSV file
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = None, decimal = ",", engine='python')

                except: 
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = ',', decimal = "." , engine='python')

            else:
                return True,'ERROR: File format not admited' , False,'',  pu.get_upload_data_component_text()

        except Exception as e:
            print(e)
            return True,'An error occurred processing the file' , False,'',  pu.get_upload_data_component_text()
        
    
        n_samples, n_features=dataframe.shape
        if(n_samples == 0):
            return True,'ERROR: The file does not contain any sample' , False,'',  pu.get_upload_data_component_text()
        elif(n_features<=2):
            return True,'ERROR: The file must contain at least 2 features' , False,'',  pu.get_upload_data_component_text()
            
        div1 = pu.div_info_loaded_file(filename,
                                    str(n_samples),
                                    str(n_features))


        file_features_dtypes = dataframe.dtypes.to_dict()

        if(len(session_data.get_features_dtypes().keys()) != len(file_features_dtypes.keys()) ):
            #dim no coincide
            error_div = html.Div(   
                            children = [
                                div1,
                                html.P('ERROR: Model dimensionality and selected Dataset ones are not the same. Please, upload a file with the same feature length') 
                            ]
                        ) 
            return True, error_div, False,'',pu.get_upload_data_component_text()

        elif(session_data.get_features_dtypes() != file_features_dtypes ):
            #dtypes no coinciden
            error_div = html.Div(   
                            children = [
                                div1,
                                html.P('ERROR: Model features-types and selected Dataset ones are not the same. Please, upload a file with the same feature-dtype as the dataset to train the model.') 
                            ]
                        ) 
            return True, error_div, False,'',pu.get_upload_data_component_text()


        else:
            #reorder selected dataframe cols to be the same as trained model
            cols = list(file_features_dtypes.keys()) 
            dataframe = dataframe[cols]

            dirpath = Path(DIR_APP_DATA)
            if( not dirpath.exists()):
                os.mkdir(dirpath)
            with open(ANOMALY_DF_PATH, 'wb') as handle:
                pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return False, '', True, div1, pu.get_upload_data_component_text()
       
    else: 
        return True,'An error occurred processing the file' , False,'',  pu.get_upload_data_component_text()
        #return  None,'' , False,  div_info_dataset( None,0) ,hidden_file_info_style, False,True, '', pu.get_upload_data_component_text()


#enable anomaly button
@app.callback(  Output('anomaly_button','disabled'),
                Input('collapse_error_loaded_file_anomaly', 'is_open'),
                Input('normality_percentage', 'value'),
                prevent_initial_call=True
)
def enable_anomaly_button(error_is_open, v):
    if(error_is_open):
        return True
    elif(v is None or v<0 or v >1):
        return True
    else:
        return False



# anomaly button click
@app.callback(  Output('collapse_anomaly_result_table', 'is_open'),
                Output('collapse_anomaly_result_table', 'children'),
                Output('anomaly_button','color'),#just for loading animaton
                Input('anomaly_button','n_clicks'),
                #State('normality_percentage_slider', 'value'),
                State('normality_percentage', 'value'),
                State('dataset_portion_radio_analyze_som','value'),
                prevent_initial_call=True
)
def detect_anomalies(n1, min_normality_percentage, data_portion_option):


    #With normal data( dataset loaded at home), we delete the neurons that are not BMU for any
    # of this data, that means hit rate is 0
    som = session_data.get_modelo() 
    model_data =  session_data.get_data(data_portion_option)
    pre_calc_freq = session_data.get_calculated_freq_map()
    print('\t--> Looking for anomalies in Loaded Data...')
    
    if(    pre_calc_freq is None or 
            (pre_calc_freq is not None and pre_calc_freq[1] != data_portion_option) ): #recalculate freq map
        frequencies = som.activation_response(model_data)
    else:#load last calculated map
        frequencies,_, _ = pre_calc_freq
  
    #filter minimum hit rate per neuron
    frequencies = ma.masked_less(frequencies, 1)
    mask = ma.getmask(frequencies) #we take the musk to apply it to weights map
    pesos = som.get_weights().copy()
    '''
    print('pesos shape', pesos.shape)
    masked_pesos = ma.masked_array(pesos, mask=mask[:,:,np.newaxis])
    masked_pesos = masked_pesos.filled(np.inf) 
    print('masked_pesos.shape',masked_pesos.shape, flush = True )
    '''
    print('\t\t--> Deleting neurons that are not BMUs for Normal Data...') #since it is a np array we just put their values to np.inf so they will never be BMU
    _,_, z = pesos.shape
    for i in range(z):
        #debug
        #if(i ==0):
        #    print('debug pesos[:,:,0]', pesos[:,:,i])
        #working

        #pesos[:,:,i] = ma.masked_array(pesos[:,:,i], mask=mask).filled(np.inf) 
        pesos[:,:,i] = ma.masked_array(pesos[:,:,i], mask=mask)


        #if(i ==0):
        #    print('debug pesos[:,:,0]', pesos[:,:,i])
        #working

    print('\t\t--> Calculating hyperparameters...')
    qes_normal_data = []
    for d in model_data:#this could be parallelized
        v_coord, h_coord = som.winner(d)
        bmu = pesos[v_coord][h_coord]
        #euclidean distance
        #print('d',d)
        #print('bmu', bmu)
        qe = np.linalg.norm(np.subtract(d, bmu), axis=-1)
        qes_normal_data.append(qe)
    n = len(qes_normal_data)
    #print('qes_normal_data',qes_normal_data)

    print('\t\t--> Detecting anormal data in loaded dataset...')
    with open(ANOMALY_DF_PATH , 'rb') as handle:
        dff = pickle.load(handle)
        test_data = session_data.estandarizar_data( dff, string_info = '', data_splitted = False)
    #controlar que la distancia no sea np.inf

    #Info table
    data = []
    columns = []
    columns_names = session_data.get_features_names()
    for col in columns_names:
        columns.append({'id': col        , 'name': col })
    columns.append({'id': 'Most Anomalous Feature'        , 'name': 'Most Anomalous Feature'  })
    columns.append({'id': 'Second Most Anomalous Feature'        , 'name': 'Second Most Anomalous Feature'  })
    columns.append({'id': 'Third Anomalous Feature'        , 'name': 'Third Most Anomalous Feature'  })



    no_std_test_data = dff.to_numpy()
    for d, no_std in zip(test_data, no_std_test_data):
        v_coord, h_coord = som.winner(d)
        bmu = pesos[v_coord][h_coord]
        qe = np.linalg.norm(np.subtract(d, bmu), axis=-1)
        b = sum(q > qe for q in qes_normal_data)
        #print('qe',qe)
        normality_value = b/n
        #print('normality_value', normality_value)
        if(normality_value<min_normality_percentage):
            #Get the top 3 most desviating features
            diferencias = np.abs(np.subtract(d, bmu))
            index_order= np.argsort(diferencias)
            first_f_index = index_order[-1]
            second_f_index = index_order[-2]
            third_f_index = index_order[-3]
            #add anomaly to table
            row = {}
            for i,c in enumerate(columns_names):
                #row[c] = d[i]
                row[c] = no_std[i] #recovered original data, since d is std
            row['Most Anomalous Feature'  ] = columns_names[first_f_index]
            row['Second Most Anomalous Feature' ] = columns_names[second_f_index]
            row['Third Anomalous Feature'] = columns_names[third_f_index]
      
            data.append(row)
            #see the top 3 features 
    print('\t\t--> Anomalies Search Finished')
    title = html.H5("Detected Potential Anomalies")
    s_table = pu.create_simple_table(data, columns, 'table_detected_anomalies')
    tablediv = html.Div(children=s_table, style = {"overflow": "scroll"})

    return  True,html.Div([title, tablediv]) , 'warning'




# show_save_anomalies_to_file_menu
@app.callback(  Output('collapse_show_save_found_anomalies_to_file', 'is_open'),
                Output('save_anomalies_showmenu_button', 'disabled'),
                Input('table_detected_anomalies', 'data'),
                prevent_initial_call=True
)
def show_save_anomalies_to_file_menu(data):
    if(data is None or len(data) == 0):
        return False, True
    else:
        return True, False

# show_ filename menu
@app.callback(  Output('collapse_filename_save_anomalies', 'is_open'),
                Input('save_anomalies_showmenu_button', 'n_clicks'),
                State('collapse_filename_save_anomalies', 'is_open'),
                prevent_initial_call=True
)
def open_filename_save_anomalies(n,is_open):
    return not is_open



#Anomaly file name valid
@app.callback(  Output('filename_anomaly_rows_input', 'valid'),
                Output('filename_anomaly_rows_input', 'invalid'),
                Output('filename_anomaly_rows_button', 'disabled'),
                Input('filename_anomaly_rows_input', 'value'),
                prevent_initial_call=True
)
def check_saveanomalies_filename(value):
    if not normpath(value) or search(r'[^A-Za-z0-9_\-]',value):
        return False,True, True
    else:
        return True,False, False



#Save csv file with anomalies results
@app.callback(  Output('correctly_saved_anomalies_rows','children'),
                Input('filename_anomaly_rows_button', 'n_clicks'),
                State('filename_anomaly_rows_input', 'valid'),
                State('filename_anomaly_rows_input', 'value'),
                State('table_detected_anomalies', 'data'),
                prevent_initial_call=True
)
def save_rows(n, valid, file_name, data):

    if(valid and data is not None and len(data)> 0):

        columns = []
        columns_names = session_data.get_features_names()
        for col in columns_names:
            columns.append(col )
        columns.append('Most Anomalous Feature'  )
        columns.append('Second Most Anomalous Feature')      
        columns.append('Third Anomalous Feature' )   

        dirpath = Path(ANOMALIES_CSV_PATH)
        if( not dirpath.exists()):
            os.mkdir(dirpath)
        filename =   file_name +  '_potencial_anomalies.csv'
      
        with open(ANOMALIES_CSV_PATH + filename, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for d in data:
                row = [ d[feature] for feature in columns_names]
                row.append(d['Most Anomalous Feature']  )
                row.append(d['Second Most Anomalous Feature'])      
                row.append(d['Third Anomalous Feature'] )  
                writer.writerow(row)

        return 'Results in Table Saved! Filename: ' + filename

    else:
        return ''