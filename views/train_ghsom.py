import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import  views.elements as elements






# Formulario GHSOM
formulario_ghsom =  dbc.ListGroupItem([
                    html.H4('Elección de parámetros',className="card-title"  ),

                    html.H5(children='Tau 1:'),
                    dcc.Slider(id='tau1_slider', min=0,max=1,step=0.00001,value=0.9),
                    dcc.Input(id="tau1", type="number", value="0.9",step=0.00001,min=0,max=1),

                    html.H5(children='Tau 2:'),
                    dcc.Input(id="tau2", type="number", value="0.5",step=0.00001,min=0,max=1),
                    dcc.Slider(id='tau2_slider', min=0,max=1,step=0.00001,value=0.5),

                    html.H5(children='Tasa de aprendizaje:'),
                    dcc.Input(id="tasa_aprendizaje", type="number", value="0.15",step=0.01,min=0,max=5),

                    html.H5(children='Decadencia:'),
                    dcc.Input(id="decadencia", type="number", value="0.95",step=0.01,min=0,max=1),   

                    html.H5(children='Sigma gaussiana:'),
                    dcc.Input(id="sigma", type="number", value="1.5",step=0.01,min=0,max=10),
                    html.Hr(),
                    html.Div( 
                        [dbc.Button("Entrenar", id="train_button_ghsom",disabled= True, className="mr-2", color="primary")],
                        style={'textAlign': 'center'}
                    )
                    


                ])






###############################   TABS     ##############################
tab1_content =  dbc.Card(color = 'light',children=[
        dbc.CardBody(
            dbc.ListGroup([
                # Dataset info
                dbc.ListGroupItem([
                    html.H4('Información del dataset:',className="card-title"  ),
                    elements.table
                ]),
                elements.model_selector,
                #Param election
                html.Div(id='modelo',children = formulario_ghsom )
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







