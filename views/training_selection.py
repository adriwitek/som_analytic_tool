import dash_core_components as dcc
import dash_html_components as html


layout = html.Div(children=[

    html.H1(children='Información del dataset:'),
    html.Table([
        html.Thead(
            html.Tr([html.Th('Número de muestras'), html.Th('Número de características')]),
        ),
        html.Tbody([
            html.Tr([html.Td(id = 'n_samples'), html.Td(id= 'n_features')]),
        ])
    ]),

    #Parametros
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

    #Grafo represnatndo al arbol ghsom
    dcc.Graph(id='ghsom_tree_structure')


])