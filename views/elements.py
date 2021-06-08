# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements


from  views.session_data import Sesion
from  config.config import *

import base64






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
        dbc.NavbarToggler(id="navbar-toggler")    
    ],
    color="white")



