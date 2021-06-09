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
import numpy.ma as ma


from  views.session_data import session_data
from  config.config import  DIR_SAVED_MODELS, UMATRIX_HEATMAP_COLORSCALE


import pickle

from  os.path import normpath 
from re import search 
import views.plot_utils as pu
from logging import raiseExceptions




##################################################################
#                       AUX FUNCTIONS
##################################################################



def info_trained_params_gsom_table():

    info = session_data.get_gsom_model_info_dict()

    
    #Table
    table_header = [
         html.Thead(html.Tr([
                        html.Th("Initial Grid Horizontal Size"),
                        html.Th("Initial Grid Vertical Size"),
                        html.Th("Tau 1"),
                        html.Th("Learning Rate"),
                        html.Th("Decadency"),
                        html.Th("Gaussian Sigma"),
                        html.Th("Epochs per Iteration"),
                        html.Th("Max Iterations"),
                        html.Th("Dissimilarity Function"),
                        html.Th("Seed")
        ]))
    ]



    if(info['check_semilla'] == 0):
        semilla = 'No'
    else:
        semilla = 'Yes: ' + str(info['seed']) 
   
    row_1 = html.Tr([html.Td( info['tam_eje_horizontal']),
                    html.Td( info['tam_eje_vertical']),
                     html.Td( info['tau_1']) ,
                     html.Td( info['learning_rate']),
                     html.Td( info['decadency']) ,
                     html.Td( info['sigma']) ,
                     html.Td( info['epocas_gsom']) ,
                     html.Td( info['max_iter_gsom']  ),
                     html.Td( info['fun_disimilitud']  ),
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
                                id="dataset_portion_radio_analyze_gsom",
                            )
                        ]
                )

#Card: Statistics
def get_statistics_card_gsom():

    return  dbc.CardBody(children=[ 
                    html.Div( id='div_estadisticas_gsom',children = '', style={'textAlign': 'center'}),
                    html.Div([
                        dbc.Button("Calculate", id="ver_estadisticas_gsom_button", className="mr-2", color="primary")],
                        style={'textAlign': 'center'}
                    )
            ])
               
            



#Card: Winners map
def get_winnersmaps_card_gsom():
              
    return  dbc.CardBody(children=[ 


                dbc.Alert(
                        [
                            html.H4("Target not selected yet!", className="alert-heading"),
                            html.P(
                                "Please select a target below to print winners map. "
                            )
                        ],
                        color='danger',
                        id='alert_target_not_selected_gsom',
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
                        id='output_alert_too_categorical_targets_gsom',
                        is_open=False

                ),


                dcc.Dropdown(id='dropdown_target_selection_gsom',
                           options=session_data.get_targets_options_dcc_dropdown_format() ,
                           multi=False,
                           value = session_data.get_target_name()
                ),
                html.Br(),    

                dbc.Collapse(
                        id='collapse_winnersmap_gsom',
                        is_open=False,
                        children=[

                            html.Div(id = 'div_winners_map_gsom',children='',
                                    style= pu.get_single_heatmap_css_style()
                            ),
                            html.Div([  
                                    dbc.Checklist(options=[{"label": "Label Neurons", "value": 1}],
                                                value=[],
                                                id="check_annotations_win_gsom"),

                                    dbc.Collapse(
                                        id='collapse_logscale_winners_gsom',
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
                                                        id="radioscale_winners_gsom",
                                                        inline=True,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),


                                    dbc.Button("Plot", id="ver_winners_map_gsom_button", className="mr-2", color="primary")],
                                style={'textAlign': 'center'}
                            )
                ])

            ])
        


#Card: Freq Map
def get_freqmap_card_gsom():

    return  dbc.CardBody(children=[ 
                html.Div(id = 'div_freq_map_gsom',children='',
                        style= pu.get_single_heatmap_css_style()
                ),

                dbc.FormGroup(
                    [
                        dbc.RadioItems(
                            options=[
                                {"label": "Linear Scale", "value": 0},
                                {"label": "Logarithmic Scale", "value": 1},
                            ],
                            value=0,
                            id="radioscale_freq_gsom",
                            inline=True,
                        ),
                    ],
                    style={'textAlign': 'center'}
                ),


                html.Div(style=pu.get_css_style_inline_flex(),
                        children = [
                            html.H6( dbc.Badge( 'Minimum hits to plot a neuron   ' ,  pill=True, color="light", className="mr-1")   ),
                            html.H6( dbc.Badge( '0',  pill=True, color="warning", className="mr-1",id ='badge_min_hits_slider_gsom')   ),
                        ]
                    ),            
                dcc.Slider(id='min_hits_slider_gsom', min=0,max=0,value=0,step =1 ),

                html.Div([  
                        dbc.Button("Plot", id="ver_freq_map_gsom_button", className="mr-2", color="primary")],
                    style={'textAlign': 'center'}
                )
            ])
               
            


#Card: 
def get_componentplans_card_gsom():

#Card: Component plans
            
    return  dbc.CardBody(children=[
                    html.H5("Select Features"),
                    dcc.Dropdown(
                        id='dropdown_atrib_names_gsom',
                        options=session_data.get_data_features_names_dcc_dropdown_format(),
                        multi=True
                    ),
                    html.Div( 
                        [dbc.Checklist(
                            options=[{"label": "Select All", "value": 1}],
                            value=[],
                            id="check_seleccionar_todos_mapas_gsom"),

                        dbc.Checklist(  options=[{"label": "Label Neurons", "value": 1}],
                                        value=[],
                                        id="check_annotations_comp_gsom"),
                        dbc.FormGroup(
                            [
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Linear Scale", "value": 0},
                                        {"label": "Logarithmic Scale", "value": 1},
                                    ],
                                    value=0,
                                    id="radioscale_cplans_gsom",
                                    inline=True,
                                ),
                            ]
                        ),

                        dbc.Button("Plot Selected Components Map", id="ver_mapas_componentes_button_gsom", className="mr-2", color="primary")],
                        style={'textAlign': 'center'}
                    ),
                    html.Div(id='component_plans_figures_gsom_div', children=[''],
                            style=pu.get_css_style_inline_flex()
                    )
                    
            ])
               

# Card freq + cplans
def get_freq_and_cplans_cards_gsom():
    children = []
    children.append(get_freqmap_card_gsom())
    children.append(html.Hr())
    children.append(get_componentplans_card_gsom())
    return html.Div(children)


          
#Card: U Matrix
def get_umatrix_card_gsom():

    return      dbc.CardBody(children=[
                    html.Div(id = 'umatrix_div_fig_gsom',children = '',
                            style= pu.get_single_heatmap_css_style()
                    ),
                    html.Div( 
                        [dbc.Checklist(  options=[{"label": "Label Neurons", "value": 1}],
                                            value=[],
                                            id="check_annotations_umax_gsom"),
                        dbc.FormGroup(
                            [
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Linear Scale", "value": 0},
                                        {"label": "Logarithmic Scale", "value": 1},
                                    ],
                                    value=0,
                                    id="radioscale_umatrix_gsom",
                                    inline=True,
                                ),
                            ]
                        ),
                            
                        dbc.Button("Plot", id="ver_umatrix_gsom_button", className="mr-2", color="primary")],
                        style={'textAlign': 'center'}
                    )
                ])

                
     

#Card: 
def get_savemodel_card_gsom():
    
           
    return  dbc.CardBody(children=[
                    html.Div(children=[
                                html.H5("Nombre del fichero"),
                                dbc.Input(id='nombre_de_fichero_a_guardar',placeholder="Nombre del archivo", className="mb-3"),
                                dbc.Button("Guardar modelo", id="save_model_gsom", className="mr-2", color="primary"),
                                html.P('',id="check_correctly_saved_gsom")
                            ],
                        style={'textAlign': 'center'}
                    ),
            ])
              







#############################################################
#	                       LAYOUT	                        #
#############################################################


def analyze_gsom_data():

    # Body
    body =  html.Div(children=[
        html.H4('Data Analysis',className="card-title"  ),

        html.H6('Train Parameters',className="card-title"  ),
        html.Div(id = 'info_table_gsom',children=info_trained_params_gsom_table(),style={'textAlign': 'center'} ),

        html.Div(children=[ 
            

            dbc.Tabs(
                id='tabs_gsom',
                active_tab='statistics',
                style =pu.get_css_style_inline_flex(),
                children=[
                    dbc.Tab(get_select_splitted_option_card(),label = 'Select Splitted Dataset Part',tab_id='splitted_part',disabled= (not session_data.data_splitted )),
                    dbc.Tab( get_statistics_card_gsom() ,label = 'Statistics',tab_id='statistics'),
                    dbc.Tab( get_winnersmaps_card_gsom() ,label = 'Winners Map',tab_id='winners_map'),
                    #dbc.Tab( get_freqmap_card_gsom() ,label = 'Freq Map',tab_id='freq_map'),
                    #dbc.Tab( get_componentplans_card_gsom() ,label = 'Component Plans',tab_id='component_plans'),
                    dbc.Tab( get_freq_and_cplans_cards_gsom(), label=' Freq. Map + Component Plans',tab_id='freq_and_cplans_gsom'),
                    dbc.Tab( get_umatrix_card_gsom() ,label = ' U Matrix',tab_id='umatrix'),
                    dbc.Tab(  get_savemodel_card_gsom() ,label = 'Save Model',tab_id='save_model'),
                ]
            ),

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


#Toggle winners map if selected target
@app.callback(
    Output('alert_target_not_selected_gsom', 'is_open'),
    Output('collapse_winnersmap_gsom','is_open'),
    Output('dropdown_target_selection_gsom', 'value'),
    Input('info_table_gsom', 'children'), #udpate on load
    Input('dropdown_target_selection_gsom', 'value'),
)
def toggle_winners_gsom(info_table,target_value):

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
        return True,False, None


    else:
        session_data.set_target_name(target_value)
        return False, True,dash.no_update



#Toggle log scale option in winners map
@app.callback(
            Output('collapse_logscale_winners_gsom', 'is_open'),
            Output('radioscale_winners_gsom','radioscale_winners_gsom'),
            Input('dropdown_target_selection_gsom', 'value'),
            State('dataset_portion_radio_analyze_gsom','value'),
            #prevent_initial_call=True 
)  
def toggle_select_logscale_gsom(target_value, option):

    if(target_value is None or not target_value):
        return False,0

    target_type,_ = session_data.get_selected_target_type( option)

    if(target_type is None or target_type == 'string'):
        return False,0

    else:
        return True,dash.no_update



#Estadisticas
@app.callback(Output('div_estadisticas_gsom', 'children'),
              Input('ver_estadisticas_gsom_button', 'n_clicks'),
              State('dataset_portion_radio_analyze_gsom','value'),
              prevent_initial_call=True )
def ver_estadisticas_gsom(n_clicks,data_portion_option):

    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    data = session_data.get_data(data_portion_option)
    gsom.replace_parent_dataset(data)

    qe, mqe = gsom.get_map_qe_and_mqe()
    params = session_data.get_gsom_model_info_dict()
    fun_disimilitud = params['fun_disimilitud']
    

  
  
    #Table
    table_header = [
        html.Thead(html.Tr([html.Th("Magnitude"), html.Th("Value")]))
    ]
    
    if(fun_disimilitud == 'qe'):
        row0 = html.Tr([html.Td(" Sumatorio de  Errores de Cuantización(neuronas)"), html.Td(qe)])
        row1 = html.Tr([html.Td("Promedio de  Errores de Cuantización(neuronas)"), html.Td(mqe)])
    else:
        row0 = html.Tr([html.Td("Sumatorio de  Errores de Cuantización Medios(neuronas)"), html.Td(qe)])
        row1 = html.Tr([html.Td("Promedio de  Errores de Cuantización Medios(neuronas)"), html.Td(mqe)])

    table_body = [html.Tbody([row0,row1])]
    table = dbc.Table(table_header + table_body,bordered=True,dark=False,hover=True,responsive=True,striped=True)
    children = [table]

    return children



#Winners map
@app.callback(Output('div_winners_map_gsom','children'),
              Output('output_alert_too_categorical_targets_gsom','is_open'),
              Input('ver_winners_map_gsom_button','n_clicks'),
              Input('check_annotations_win_gsom','value'),
              Input('radioscale_winners_gsom','value'),
              State('dataset_portion_radio_analyze_gsom','value'),
              prevent_initial_call=True 
              )
def update_winner_map_gsom(click,check_annotations,logscale, data_portion_option):


    output_alert_too_categorical_targets_gsom = False
    log_scale = False


    data = session_data.get_data(data_portion_option)
    targets_list = session_data.get_targets_list(data_portion_option)

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
            positions[(r,c)].append(targets_list[i]) 


        else:
            positions[(r,c)] = []
            positions[(r,c)].append(targets_list[i]) 



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
                if((i,j) in positions):
                        data_to_plot[i][j]  = np.mean(positions[(i,j)])
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
            output_alert_too_categorical_targets = True


        #showing the class more represented in each neuron
        for i in range(tam_eje_vertical):
            for j in range(tam_eje_horizontal):
                if((i,j) in positions):
                        c = Counter(positions[(i,j)])
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
                                     text = text, discrete_values_range= values, unique_targets = unique_targets,log_scale=log_scale)
    if(table_legend is not None):
        children = pu.get_fig_div_with_info(fig,'winners_map_gsom', 'Winning Neuron Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,
                                            neurona_padre=None,  table_legend =  table_legend)
    else:
        children = pu.get_fig_div_with_info(fig,'winners_map_gsom', 'Winning Neuron Map',tam_eje_horizontal, tam_eje_vertical,gsom_level= None,neurona_padre=None)

    print('\n GSOM Winning Neuron Map: Plotling complete! \n')
    return children, output_alert_too_categorical_targets_gsom


   

#update_selected_min_hit_rate_badge_gsom
@app.callback(  Output('badge_min_hits_slider_gsom','children'),
                Input('min_hits_slider_gsom','value'),
                prevent_initial_call=True 
)
def update_selected_min_hit_rate_badge_gsom(value):
    return int(value)



#Frequency map
@app.callback(  Output('div_freq_map_gsom','children'),
                Output('min_hits_slider_gsom','max'),
                Output('min_hits_slider_gsom','marks'),
                Input('ver_freq_map_gsom_button','n_clicks'),
                Input('radioscale_freq_gsom','value'),
                Input('min_hits_slider_gsom','value'),
                State('dataset_portion_radio_analyze_gsom','value'),
                prevent_initial_call=True 
)
def update_freq_map_gsom(click,logscale,slider_value, data_portion_option):

    data = session_data.get_data(data_portion_option)
    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()
    pre_calc_freq = session_data.get_calculated_freq_map()


    if( pre_calc_freq is None or (pre_calc_freq is not None and pre_calc_freq[1] != data_portion_option) ): #recalculate freq map
        
        data_to_plot = np.zeros([tam_eje_vertical ,tam_eje_horizontal],dtype=int)
        # Getting winnig neurons for each data element
        for i,d in enumerate(data):
            winner_neuron = gsom.winner_neuron(d)[0][0]
            r, c = winner_neuron.position
            data_to_plot[r][c] = data_to_plot[r][c] + 1


        session_data.set_calculated_freq_map(data_to_plot,data_portion_option )

    else:#load last calculated map
        data_to_plot,_, _ = pre_calc_freq

    max_freq = np.nanmax(data_to_plot)
    if(max_freq > 0):
        marks={
            0: '0 hits',
            int(max_freq): '{} hits'.format(int(max_freq))
        }
    else:
        marks = dash.no_update

     
    if(slider_value != 0):#filter minimum hit rate per neuron
        #frequencies = np.where(frequencies< slider_value,np.nan,frequencies)
        data_to_plot = ma.masked_less(data_to_plot, slider_value)
        session_data.set_freq_hitrate_mask(ma.getmask(data_to_plot))
    else:
        session_data.set_freq_hitrate_mask(None)



    fig,_ = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,True, 
                                     title = None,log_scale = logscale)
    children = pu.get_fig_div_with_info(fig,'freq_map_gsom', 'Frequency Map',tam_eje_horizontal, tam_eje_vertical)

    return children, max_freq,marks





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
              Input('min_hits_slider_gsom','value'),
              State('dropdown_atrib_names_gsom','value'),
              State('check_annotations_comp_gsom','value'),
              State('radioscale_cplans_gsom','value'),
              prevent_initial_call=True 
              )
def update_mapa_componentes_gsom_fig(n_cliks,slider_value ,names, check_annotations, log_scale):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if(n_cliks == 0 and trigger_id == 'min_hits_slider_gsom' ):
        raise PreventUpdate
    elif(trigger_id == 'min_hits_slider_gsom' and ( names is None or  len(names)==0)):
        return dash.no_update


    #params = session_data.get_gsom_model_info_dict()
    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
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
            
        id ='graph-{}'.format(k)
        figure,_ = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations,
                                             title = nombres_atributos[k], log_scale = log_scale)
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
        atribs = session_data.get_features_names()
        return atribs
    else:
        return []






#Ver UMatrix GSOM
@app.callback(Output('umatrix_div_fig_gsom','children'),
              Input('ver_umatrix_gsom_button','n_clicks'),
              Input('check_annotations_umax_gsom','value'),
              Input('radioscale_umatrix_gsom','value'),
              prevent_initial_call=True 
              )
def ver_umatrix_gsom_fig(click, check_annotations, log_scale):

    #print('Button clicked, calculating umatrix')
    #params = session_data.get_gsom_model_info_dict()
  
    zero_unit = session_data.get_modelo()
    gsom = zero_unit.child_map
    tam_eje_vertical,tam_eje_horizontal=  gsom.map_shape()

    weights_map= gsom.get_weights_map()
    # weights_map[(row,col)] = np vector whith shape=n_feauters, dtype=np.float32

    data_to_plot = np.empty([tam_eje_vertical ,tam_eje_horizontal],dtype=np.float64)
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
    fig,_ = pu.create_heatmap_figure(data_to_plot,tam_eje_horizontal,tam_eje_vertical,check_annotations, title = None,
                                    colorscale = UMATRIX_HEATMAP_COLORSCALE,  reversescale=True,  log_scale = log_scale)
    children =  pu.get_fig_div_with_info(fig,'umatrix_fig_gsom', 'U-Matrix',tam_eje_horizontal, tam_eje_vertical)

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
    columns_dtypes = session_data.get_features_dtypes()

    data.append('gsom')
    data.append(columns_dtypes)
    data.append(params)
    data.append(session_data.get_modelo())

    filename =   name +  '_gsom.pickle'

    with open(DIR_SAVED_MODELS + filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 'Model saved! Filename: ' + filename