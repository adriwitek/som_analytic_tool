# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements


'''
    Common visual  elements for all views
'''


#Global Var.
APP_NAME = 'SOM Analytic Tool'
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"







# Html elements
cabecera = html.Div(className='jumbotron'  ,children=[
    html.H1(children=APP_NAME),
    html.P(children = 'Herramienta de análisis de datos con Mapas Auto-organizados'),
    html.Hr()
])


navigation_bar = dbc.Navbar(
[           
        # To store session variables
        dcc.Store(id='session_data', storage_type='session'),
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
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