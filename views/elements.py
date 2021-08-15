# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_html_components.A import A
import  views.elements as elements


from  views.session_data import Sesion
from  config.config import *

import base64
from views.app import app
from dash.dependencies import Input, Output, State
import views.plot_utils as pu






# Html elements
cabecera =  dbc.Collapse(   id = 'cabecera',
                            is_open = True,
                            children = 

                                html.Div(className='jumbotron'  ,
                                                    children=[
                                                    
                                                        html.Div(id="hidden_div_for_redirect_callback"),
                                                        html.H1(children=APP_NAME,style = {'font-size':'4vw'}),
                                                        html.P(children = 'Graphic Interactive Tool for Data Analysis and Visualization with Self Organized Maps'),
                                                        html.Hr()
                                ])
            )




navigation_bar = dbc.Navbar(
[           
        # To store session variables
        dcc.Store(id='session_data', storage_type='session'),
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    #dbc.Col(html.Img(src= base64.b64encode(open(CONFIG_LOGO, 'rb').read()), height="30px")),
                    dbc.Col(dbc.NavbarBrand(APP_NAME, className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="/",
        ),

        dbc.Modal(
            [
                dbc.ModalHeader("About"),
                dbc.ModalBody([
                    html.Div(style=pu.get_css_style_inline_flex(),
                            children = [
                                html.H6( dbc.Badge( 'Developer',  pill=True, color="light", className="mr-1",id ='badge_info_percentage_train_slider')   ),
                                html.H6( dbc.Badge( 'adriwitek',  pill=True, color="info", className="mr-1",id ='badge_info_percentage_test_slider')   )
                            ]
                    ),
                    dbc.Button('https://github.com/adriwitek', color="link",href = 'https://github.com/adriwitek' ),
                    html.Br(),
                    html.Br(),
                    dbc.Button("Close", id="button_close_about", className="ml-auto"),
                    ],style = pu.get_css_style_center()
                ),
            ],
            id="modal_about",
            centered=True,
            is_open= False,
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("About", id= 'button_about',color="link"),
                    width="auto",
                ),
            ],
            no_gutters=True,
            className="ml-auto flex-nowrap mt-3 mt-md-0",
            align="center",
        ),
                
        dbc.NavbarToggler(id="navbar-toggler")   
    ],
    color="white")



@app.callback(  Output("modal_about", "is_open"),
                Input("button_about", "n_clicks"), 
                Input("button_close_about", "n_clicks"), 
                State("modal_about", "is_open"),
                prevent_initial_call=True
)
def show_about_modal( n_clicks1, n_clicks2, is_open):
    return (not is_open)