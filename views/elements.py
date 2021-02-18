# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

'''
    Common visual  elements for all views
'''

APP_NAME = 'SOM Analytic Tool'
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"


cabecera = html.Div(className='jumbotron'  ,children=[
    html.H1(children=APP_NAME),
    html.P(children = 'Herramienta de análisis de datos con Mapas Auto-organizados'),
    html.Hr()
])

'''
navigation_bar = html.Nav(className='navbar navbar-expand-lg navbar-light bg-light'  ,children=[
    html.A(className='navbar-brand', children=APP_NAME ,href='/'),

    html.Li(className='nav-item dropdown' ,
        children= [
        html.A(className='nav-link dropdown-toggle', children='Modelos' ,role = 'button'),
        html.Div(className='dropdown-menu' ,children = [
            html.A(className='dropdown-item', children='SOM' ,href='#'),
            html.A(className='dropdown-item', children='GHSOM' ,href='#')
        ])
    ])
])
'''

navigation_bar = dbc.Navbar(
[
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