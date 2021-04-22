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
                        html.H4('Entrenando...',className="card-title" , style={'textAlign': 'center'} ),


                        dcc.Interval(id="progress_interval", n_intervals=0, interval=500, disabled = False),
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
            Input("progress_interval", "n_intervals"),
)
def update_progress(n):
 

    progress = session_data.get_progressbar_value()
    '''
    if(progress is None):
       return 0, '', False ,True, ''
    '''
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    #progress = min(n % 110, 100)


    # only add text after 5% progress to ensure text isn't squashed too much
    if(int(progress) == 100 ):
        return progress, f"{int(progress)} %",True, False, session_data.get_current_model_type_analyze_url()
    else:
        return progress, f"{round(progress,2)} %" if progress >= 5 else "",False, True,''


