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

import time



#############################################################
#	                       LAYOUT	                        #
#############################################################


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
                        html.H4('Entrenando...', id = 'status_string', className="card-title" , style={'textAlign': 'center'} ),
                        html.Div(children =dbc.Badge("Tiempo transcurrido:", id ='badge_t_transcurrido', color="warning", className="mr-1"),
                                style={'textAlign': 'center'}  
                        ),
                        html.P(id='timer_training',children="00 h 00 m 00 s", className="text-muted",  style= {'font-family': 'Courier New',  'font-size':'2vw', 'textAlign': 'center' }),



                        dcc.Interval(id="progress_interval", n_intervals=0, interval=1000, disabled = False),
                        dbc.Progress(id="progressbar" ,value = 0),

                        
                        html.Div( 
                            [dbc.Button("Analizar Modelo", id="analyze_model_button",href='',disabled= True, className="mr-2", color="primary")],
                            style={'textAlign': 'center'}
                        ),
                       




                
                    ]),
                ],flush=True,),

            )
        ]),


    ])


    return layout









@app.callback(Output("progressbar", "value"), 
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
        return progress, f"{int(progress)} %",True, False, session_data.get_current_model_type_analyze_url(),t_formatedo, 'Entrenamiento Finalizado', 'success'
    else:
        return progress, f"{round(progress,2)} %" if progress >= 5 else "",False, True,'', t_formatedo, 'Entrenando...','warning'


