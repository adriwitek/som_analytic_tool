import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements



###############################   TABLE     ##############################
table_header = [
    html.Thead(html.Tr([html.Th("Número de muestras"), html.Th("Número de características")]))
]
row1 = html.Tr([html.Td(id = 'tttttttttttttttttt',children = html.Div(id='table_info_n_samples' ), ), html.Td(id= 'table_info_n_features')])
table_body = [html.Tbody([row1])]

table = dbc.Table(table_header + table_body, bordered=True)


###############################   TABS     ##############################
tab1_content =  dbc.Card(color = 'light',children=[
        dbc.CardBody(
            dbc.ListGroup([
                # Dataset info
                dbc.ListGroupItem([
                    html.H4('Información del dataset:',className="card-title"  ),
                    table
                ]),
                #Param election
                dbc.ListGroupItem([
                    html.H4('Elección de parámetros',className="card-title"  ),
                    html.H3(children='Tau 1:'),
                    dcc.Input(id="tau1", type="number", placeholder="0.9",step=0.00001,min=0,max=1),
                    dcc.Slider(id='tau1_slider', min=0,max=1,step=0.00001,value=0.9),
                    html.H3(children='Tau 2:'),
                    dcc.Input(id="tau2", type="number", placeholder="0.5",step=0.00001,min=0,max=1),
                    html.H3(children='Tasa de aprendizaje:'),
                    dcc.Input(id="tasa_aprendizaje", type="number", placeholder="0.5",step=0.00001,min=0,max=1),
                    html.H3(children='Decadencia:'),
                    dcc.Input(id="decadencia", type="number", placeholder="0.95",step=0.00001,min=0,max=10),          
                    html.H3(children='Sigma gaussiana:'),
                    dcc.Input(id="sigma", type="number", placeholder="1.5",step=0.00001,min=0,max=10),
                    html.Button('Entrenamiento',id='train-button'),
                ]),
            ],flush=True,),

    
        )
    ])


tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Tab 1"),
        dbc.Tab(tab2_content, label="Tab 2"),
        dbc.Tab(
            "This tab's content is never seen", label="Tab 3", disabled=True
        ),
    ]
)



###############################   LAYOUT     ##############################
layout = html.Div(children=[

    elements.navigation_bar,
    tabs,
])





