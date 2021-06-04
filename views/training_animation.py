# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements
from  views.session_data import session_data
from math import trunc
import views.plot_utils as pu
import plotly.graph_objects as go

import time



#############################################################
#	                       LAYOUT	                        #
#############################################################


#Card QE evolution Plot
def get_plot_qe_evolution():
    
    if(session_data.get_show_error_evolution() ):
        fig = pu.create_qe_progress_figure([],[], session_data.get_total_iterations() )
        open = True

    else:
        fig = {}
        open = False

    children = [
                    #html.H3('Map qe Error Evolution'),
                    dcc.Graph(id='qe_evolution_figure',figure=fig) 
    ]
    collapse = dbc.Collapse(id='collapse_plot_qe_evolution',children = children, is_open = open)

    return collapse
    



def Training_animation(): 

    session_data.reset_progressbar_value()
    layout = html.Div(children=[

        html.Div(id="hidden_div_for_redirect_callback"),

        elements.navigation_bar,

    
    
        dbc.Card(color = 'light',children=[

            dbc.CardBody(
                dbc.ListGroup([

                    # Modelos guardados en la app
                    dbc.ListGroupItem([
                        html.H4('Training...', id = 'status_string', className="card-title" , style= pu.get_css_style_center()),
                        html.Div(children =dbc.Badge("Elapsed Time:", id ='badge_t_transcurrido', color="warning", className="mr-1"),
                                style={'textAlign': 'center'}  
                        ),
                        html.P(id='timer_training',children="00 h 00 m 00 s", className="text-muted",  style= {'font-family': 'Courier New', 
                                     'font-size':'2vw', 'textAlign': 'center' }),

                        html.Div(children =get_plot_qe_evolution(), style= {'textAlign': 'center' } ),

                        dcc.Interval(id="progress_interval", n_intervals=0, interval=1000, disabled = False),
                        dbc.Progress(id="progressbar" ,value = 0, color="warning"),


                        dbc.Collapse(   id='collapse_multiple_trainings',
                                        is_open = False,
                                        children = [
                                            html.Div(children = 
                                                [ 
                                                    dbc.Label('Training model '),
                                                    dbc.Badge('0',id = 'badge_current_model_training', pill=True, color="light", className="mr-1"),
                                                    dbc.Label('of '),
                                                    dbc.Badge(session_data.get_n_of_models(), pill=True, color="info", className="mr-1"),
                                                ],
                                                style = pu.get_css_style_inline_flex(),
                                            ),
                                            
                                            dbc.Progress(id="progressbar_total_trainings" , value = 0),
                                        ]
                        ),


                        html.Br(),
                        html.Div( 
                            [dbc.Button("Analyze Model", id="analyze_model_button",href='',disabled= True, className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),

                    ]),
                ],flush=True,),

            )
        ]),


    ])


    return layout









@app.callback(  Output("progressbar", "value"), 
                Output("progressbar", "children"),
                Output("progress_interval", "disabled"),
                Output("analyze_model_button", "disabled"),
                Output("analyze_model_button", "href"),
                Output("timer_training", "children"),
                Output("status_string", "children"),
                Output("badge_t_transcurrido", "color"),
                Input("progress_interval", "n_intervals"),
)
def update_progress(n):
 

    progress = session_data.get_progressbar_value()
    t_transcurrido = session_data.get_training_elapsed_time()
    t_formatedo = time.strftime("%H h %M m %S s", time.gmtime(t_transcurrido))

    # only add text after 5% progress to ensure text isn't squashed too much
    if(int(progress) == 100 ):
        cond = session_data.get_som_multiple_trains()
        if( cond):
            current_model = session_data.get_current_training_model()
            n_of_models = session_data.get_n_of_models()
            if(current_model +1 == n_of_models ):
                return progress, f"{int(progress)} %",True, False, session_data.get_current_model_type_analyze_url(),t_formatedo, 'Training Completed', 'success'
            else:#New model starts to train, reset time counter and this progress bar
                t_formatedo = time.strftime("%H h %M m %S s", time.gmtime(0))
                return 0,"", False, True, '',t_formatedo   , dash.no_update,dash.no_update
        else:
            return progress, f"{int(progress)} %",True, False, session_data.get_current_model_type_analyze_url(),t_formatedo, 'Training Completed', 'success'
    else:
        return progress, f"{round(progress,2)} %" if progress >= 5 else "",False, True,'', t_formatedo, 'Training...','warning'





@app.callback(  Output('progressbar_total_trainings','value'),
                Output("progressbar_total_trainings", "children"),
                Output('collapse_multiple_trainings','is_open'),
                Output('badge_current_model_training','children'),
                Input("progress_interval", "n_intervals"),
                Input('status_string','children' ),
)
#Callback multiple trainings - current model bar
def update_current_model_pbar(iinput, status_string):

    cond = session_data.get_som_multiple_trains()
    if(not cond):
        raise PreventUpdate

    current_model = session_data.get_current_training_model()
    n_of_models = session_data.get_n_of_models()


    if(current_model +1  == n_of_models  and status_string == 'Training Completed'):
        return 100, f"{100} %" , True, current_model +1 
    else:
        progress = (current_model*100) / n_of_models 
        # only add text after 5% progress to ensure text isn't squashed too much
        return progress, f"{round(progress,2)} %" if progress >= 5 else "", True, current_model +1 




@app.callback(  Output("qe_evolution_figure", "figure"), 
                Output("collapse_plot_qe_evolution", "is_open"), 
                Input("progress_interval", "n_intervals"),
)
def update_error_evolution(iinput):
    cond = session_data.get_show_error_evolution()
    if(cond):
        x, y = session_data.get_error_evolution()
        fig = pu.create_qe_progress_figure(x ,y , session_data.get_total_iterations() )
        return fig,True
    else:
        raise PreventUpdate
        #return {},False
    #return dcc.Graph(id='qe_evolution_figure',figure=fig) 




