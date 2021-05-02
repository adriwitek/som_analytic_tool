# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

from views.app import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import  views.elements as elements


import io
from io import BytesIO
from datetime import datetime
import base64

import pandas as pd
import numpy as np

from  views.session_data import session_data
from  config.config import *
from  config.config import DEFAULT_DATASET_ROWS_PREVIEW


import plotly.graph_objects as go

from os import listdir,makedirs
from os.path import isfile, join
import pickle



show_file_info_style =  {'textAlign': 'center',  'display': 'block'}
hidden_div_style ={'textAlign': 'center', "visibility": "hidden",'display':'none'} 

#############################################################
#	                       LAYOUT	                        #
#############################################################



#### TRAINING SELECTION
def Training_selection(): 

    layout = html.Div(id = 'training_selection_div',
        children=[

            #html.Div(id="hidden_div_for_redirect_callback"),

            #elements.navigation_bar,
            html.Div(   id='select_button', 
                        style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'},
                        children=[
                            dbc.Button("Train New Model", id="train_new_model_button", className="mr-2", color="primary", outline=True ),
                            dbc.Button("Load Pre-Trained Model", id="load_saved_model_button", className="mr-2", color="primary", outline=True  )
                        ]
            ),

            dbc.Collapse(id = 'train_newmodel_collapse',
                        is_open = False,
                        children = [
            #html.Div(id = 'train_newmodel_div',
            #            style=hidden_div_style,
            #            children = [
                        
                                dbc.Card(color = 'light',
                                    children=[
                                        dbc.CardHeader(html.H2('Train New Model')),

                                        dbc.CardBody(
                                            dbc.ListGroup([
                                            
                                                #SOM
                                                dbc.ListGroupItem([
                                                    html.H4('SOM',style={'textAlign': 'center'} ),
                                                    html.Div( 
                                                        [dbc.Button("SOM", id="train_mode_som_button", className="mr-2", color="primary",)],
                                                        style={'textAlign': 'center'}
                                                    )
                                                ]),


                                                #GSOM
                                                dbc.ListGroupItem([
                                                    html.H4('GSOM',style={'textAlign': 'center'} ),
                                                    html.Div( 
                                                        [dbc.Button("GSOM", id="train_mode_gsom_button", className="mr-2", color="primary",)],
                                                        style={'textAlign': 'center'}
                                                    )
                                                ]),


                                                #GHSOM
                                                dbc.ListGroupItem([
                                                    html.H4('GHSOM',style={'textAlign': 'center'} ),
                                                    html.Div( 
                                                        [dbc.Button("GHSOM", id="train_mode_ghsom_button", className="mr-2", color="primary",)],
                                                        style={'textAlign': 'center'}
                                                    )
                                                ]),

                                            ],flush=True,),

                                        )
                                ])

                        ]
                ), 


            dbc.Collapse(id = 'loadmodel_collapse',
                        is_open = False,
                        children = [
            #html.Div(id = 'loadmodel_div',
            #        style=hidden_div_style ,
            #        children = [
                        dbc.Card(color = 'light',
                            children=[
                                dbc.CardHeader(html.H2('Load Pre-Trained Model')),
    
                                dbc.CardBody(
                                    dbc.ListGroup([
                                    
                                        # Modelos guardados en la app
                                        dbc.ListGroupItem([
                                            html.H4('Saved Models',className="card-title" , style={'textAlign': 'center'} ),
                                            html.Div(style={'textAlign': 'center'},
                                                    children=[
                                                        dcc.Dropdown(
                                                            id='modelos_guardados_en_la_app_dropdown',
                                                            options=get_app_saved_models(),
                                                            value='',
                                                            searchable=False
                                                            #,style={'width': '35%'}
    
                                                        ),



                                                        dbc.Alert(  id="alert_load_model",
                                                                    children = "ERROR",
                                                                    color="danger",
                                                                    dismissable=False,
                                                                    is_open=False,
                                                        ),

                                                        dbc.Collapse(   id = 'info_selected_model',
                                                                        is_open = False,
                                                                        children =''
                                                        ),
    
                                                        dbc.Button("Load Selected Model", id="load_model_buton",disabled= True, className="mr-2", color="primary"),
                                                        html.Div(id='hidden_div_for_load_model',style={'textAlign': 'center'} ),
    
    
                                                    ]
    
    
                                            )
    
    
    
    
    
    
                                        ]),
                                    ],flush=True,),
    
                                )
                            ]
                        )     
    
                    ]
            ) 



    ])


    return layout



######### HOME #########
def Home(): 

    session_data.clean_session_data()
    
    layout = html.Div(children=[

        html.Div(id="hidden_div_for_redirect_callback"),

        #Dash componets for storing data
        dcc.Store(id='original_dataframe_storage',data=None),
        dcc.Store(id='processed_dataframe_storage',data=None),
        dcc.Store(id='notnumeric_dataframe_storage',data=None),
        dcc.Store(id='trainready_dataframe_storage',data=None),


        elements.navigation_bar,
        elements.cabecera,

        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Select Dataset')),

            dbc.CardBody(
                dbc.ListGroup([
                    # Archivo Local
                    dbc.ListGroupItem([
                        html.H4('Local File',className="card-title" , style={'textAlign': 'center'} ),
                        dcc.Upload( id='upload-data', children=html.Div(['Drag and Drop or  ', html.A('Select File  (.csv or .xls)')]),
                                            style={'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px'},
                                            # Allow multiple files to be uploaded
                                            multiple=False),


                        # Preview Table
                        html.Div(id = 'preview_table' ,children =''),

                        #info showed when the dataset its loaded
                        dbc.Collapse(id='collapse_modify_data_button', is_open = False, children = 
                            dbc.Button("Modify Data",id="modify_data_button",className="mb-6",color="primary",block=True)
                        ),
                        
                        dbc.Collapse(id ='info_dataset_collapse',

                            
                            children = [   
                                html.Div(id='info_dataset_div',
                                        style=hidden_div_style,
                                        children = div_info_dataset('','', '', '', None ) 
                                )
                            ]
                        )

                       

                    ]),

                 
                    
                ],flush=True,),


            )
        ]),



        #Training Selection Card
        dbc.Collapse(id = 'collapse_traing_sel_home',
            is_open= False,
            children = [
                dbc.Card( color = 'light',
                         children=[

                        dbc.CardBody(
                                html.Div(Training_selection())
                        )
                ])
            ]
        )



    ])

    return layout









#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################


def get_app_saved_models():

    makedirs(DIR_SAVED_MODELS, exist_ok=True)

    onlyfiles = [f for f in listdir(DIR_SAVED_MODELS) if isfile(join(DIR_SAVED_MODELS, f))]

    options = []  # must be a list of dicts per option
    for f in onlyfiles:
        if f.endswith('.pickle'):
            options.append({'label' : f, 'value': f})
    return options



 
def div_info_dataset(filename,fecha_modificacion, n_samples, n_features, df):
    if(df is None):
        return ''
    return html.Div(id='output_uploaded_file',children=[
                html.P(children= 'File:  ' + filename, style={'textAlign': 'center'} ),
                html.P(children= 'Last modified: ' + fecha_modificacion, style={'textAlign': 'center'} ),
                html.P(children= 'Number of Samples:  ' + n_samples , style={'textAlign': 'center'} ),
                html.P(children= 'Number of Features:  ' + n_features , style={'textAlign': 'center'} ),


                html.Div(id='output-data-upload_1',style={'textAlign': 'center'} ),
                html.Div(id='output-data-upload_2',style={'textAlign': 'center'} ),
                html.Div(id='n_samples',style={'textAlign': 'center'} ),
                html.Div(id='n_features',style={'textAlign': 'center'} ),
            




                #Atrib names
                html.H6('Feature Selection:'
                       
                ),

                dcc.Dropdown(id='dropdown_col_df_names',
                           options=[],
                           multi=True,
                           value = []
                ),


                dbc.Checklist(
                    options=[
                        {"label": "Detect and Clean Categorical Data", "value": 1},
                    ],
                    value=[],
                    id="clean_data_switch",
                    switch=True,
                ),


                dbc.Checklist(  options=[{"label": "Apply One Hot Encoding", "value": 0}],
                                            value=[],
                                            id="check_onehot"),


                html.Div(id='div_onehot_menu',children=get_onehot_childrendiv_menu()),



            
                
                #html.Div(id = 'preview_table' ,children =create_preview_table(df))
                #html.Div(id = 'preview_table' ,children =''),

                html.Div(
                    [
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Data Loaded"),
                                dbc.ModalBody("Data set has been loaded succesfully."),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="close", className="ml-auto")
                                ),
                            ],
                            id="modal",
                            centered=True,
                            is_open= False,
                        ),
                    ]
                )

                

            ])
                      



    


def create_preview_table(df):

    #TODO BORRAR ESTO SI NO QUIERO QUE SEA OBLIGATORI EL TARGET
    selected_columns = []
    
    if (len(df.columns) > 0):
        selected_columns.append(df.columns[len(df.columns)-1])
    
   
    return html.Div([
                    html.H4('Table Preview',className="card-title" , style={'textAlign': 'center'} ),

                    dcc.Loading(id='loading',
                                type='circle',
                                children=[
                                    
                                    html.Div(style = {"overflow": "scroll"},
                                        children=
                                        dash_table.DataTable(
                                            id='dataset_table_preview',
                                            column_selectable="single",
                                            columns=[{"name": i, "id": i, "selectable": True} for i in df.columns],
                                            
                                            selected_columns=selected_columns,
                                            data=df.head(DEFAULT_DATASET_ROWS_PREVIEW).to_dict('records'),

                                            style_cell={'textAlign': 'center',
                                                        'textOverflow': 'ellipsis',
                                                        'overflowX': 'auto'
                                            },



                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': 'rgb(248, 248, 248)'
                                                }
                                            ],

                                            style_header={
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'fontWeight': 'bold'
                                            }
                                        )

                                    )


                                ]
                                    
                                
                    )

    ])


def get_onehot_childrendiv_menu():

    collapse = dbc.Collapse(id="collapse_onehot_menu",children=
        dbc.CardBody(children=[ 
            html.Div(

                children = [
                
                
                    html.H5("Select features to apply One Hot:"),
                    dcc.Dropdown(
                        id='dropdown_atrib_names_home',
                        options=[],
                        multi=True
                    ),

                    dbc.Checklist(  options=[{"label": "Consider empty values", "value": 0}],
                                                        value=[],
                                                        id="check_nanvalues_onehot"
                    ),

                    dbc.Button("Apply One Hot", id="apply_onehot_button", className="mr-2", color="primary")


                ]
            )
        ])
    )

    return collapse




def getNumericVars(df, clean_categorical_data):
    #Only numeric variables
    df_numeric = df.select_dtypes(['number'])

    if(clean_categorical_data):
        # Auto detect categorical variables to exclude them
        allvars = set(df_numeric.columns)
        intvars = df_numeric.columns[df_numeric.dtypes == 'int']
        categorical = []
        for col in intvars:
            if(len(df_numeric[col].unique())<15):
                categorical.append(col)

        catvars = set(categorical)
        numericvars = allvars-catvars
        df_numeric = df_numeric[numericvars]
        df_categorial = df_numeric[catvars]
    else:
        df_categorial = pd.DataFrame(columns= [])


    df_not_numeric = df.select_dtypes(exclude = 'number')
    #print(df_not_numeric)
    frames = [df_not_numeric, df_categorial ]
    df_not_numeric = pd.concat(frames, axis=1)

    return df_numeric, df_not_numeric











#############################################################
#	                     CALLBACKS	                        #
#############################################################


#Seleccionar columna
@app.callback(
    Output('dataset_table_preview', 'style_data_conditional'),
    Input('dataset_table_preview', 'selected_columns')
)
def update_styles(selected_columns):
    #print('selected coluss', selected_columns)
    dataset_table_preview =  [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

    return dataset_table_preview
    #TODO AQUI VA LA CONDICION DE DESHABILITAR BOTON DE CARGAR MODELO SI NO HAY TARGET....
    '''
    if(not selected_columns):
        return dataset_table_preview,
    else:
        return dataset_table_preview
    '''



@app.callback(  Output('dropdown_atrib_names_home', 'options'),
                Input('notnumeric_dataframe_storage', 'data'),
                prevent_initial_call=True 
)
def onehot_dropdown_options(input_data):

    options = []  # must be a list of dicts per option
    if input_data is not None:
        dff = pd.read_json(input_data,orient='split')
        col = dff.columns
         
        for n in col:
            options.append({'label' : n, 'value': n})
    return options




#Process data
@app.callback(  Output('processed_dataframe_storage','data'),
                Output('notnumeric_dataframe_storage','data'),
                Input('original_dataframe_storage','data'),
                Input('clean_data_switch', 'value'),    
                Input('apply_onehot_button', 'n_clicks'),
                State('processed_dataframe_storage','data'),
                State('notnumeric_dataframe_storage','data'),
                State('dropdown_atrib_names_home','value'),
                State('check_nanvalues_onehot','value'),
                prevent_initial_call=True
            )
#TODO MEMORIZE
def process_data(input_data , clean_categorical_data, n_clicks, processed_data, notnum_data, names,check_nan):


    if input_data is not None:

        ctx = dash.callback_context
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        #contexto de apply onehot: modificamos processed y notnumdata
        if(button_id == 'apply_onehot_button'): 

            notnum_df   = pd.read_json(notnum_data,orient='split')

            
            if( (processed_data is None) or (not names) or (notnum_df.empty ) ):
               #devolvemos tal cual pq no hay cambios
               return  processed_data,notnum_data


            else:#applicamos onehot

                processed_df = pd.read_json(processed_data,orient='split')

                for n in names:
                    # use pd.pd.concat to join the new columns with your original dataframe
                    processed_df = pd.concat( [pd.get_dummies(notnum_df[n], prefix=str(n), dummy_na=check_nan), processed_df ],axis=1)
                    notnum_df.drop([n],axis=1, inplace=True)


                processed_data = processed_df.to_json(date_format='iso',orient = 'split')
                notnum_data = notnum_df.to_json(date_format='iso',orient = 'split')
                return  processed_data,notnum_data
       

       #contexto del switch o del original_Dataframe: procesamos datos
        else:  

            dff = pd.read_json(input_data,orient='split')
            dff_num, dff_not_num = getNumericVars(dff,bool(clean_categorical_data))

            processed_data = dff_num.to_json(date_format='iso',orient = 'split')
            notnum_data = dff_not_num.to_json(date_format='iso',orient = 'split')

            return  processed_data,notnum_data

    
    else:
        dff = pd.DataFrame(columns=[])
        return dff,dff






#Show One Hot Encoding Menu
@app.callback(Output('collapse_onehot_menu', 'is_open'),
              Input('check_onehot', 'value'),
              prevent_initial_call=True )
def show_onehot_menu(check_onehot):
    if(check_onehot):
        return True
    else:
        return False





#Carga la info del dataset en el home
@app.callback(Output('original_dataframe_storage', 'data'),
              Output('info_dataset_div', 'children'),
              Output('info_dataset_div', 'style'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'), 
              prevent_initial_call=True)
#TODO
#@cache.memoize(timeout=60)  # in seconds
def update_output( contents, filename, last_modified):
    '''Carga el dataset en los elementos adecuados

    '''
    if contents is not None:
        
        show_file_info_style =  {'textAlign': 'center',  'display': 'block'}

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                try:
                    # Assume that the user uploaded a CSV file
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = None, decimal = ",", engine='python')

                except: 
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = ',', decimal = ".")

            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                dataframe = pd.read_excel(io.BytesIO(decoded))

            else:
                return None,html.Div([ 'ERROR: File format not admited']),show_file_info_style

        except Exception as e:
            print(e)
            return None,html.Div([ 'An error occurred processing the file']), show_file_info_style
        
    

        n_samples, n_features=dataframe.shape
        if(n_samples == 0):
            return None, html.Div([ 'ERROR: The file does not contain any  sample']),show_file_info_style
        elif(n_features<=2):
            return None,html.Div([ 'ERROR: The file must contain at least 2 features']),show_file_info_style
            

      
        divv = div_info_dataset(filename,
                                datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S'),
                                str(n_samples),
                                str(n_features),
                                dataframe) 
        return dataframe.to_json(date_format='iso',orient = 'split'),divv, show_file_info_style
                

    else: 
        return  None,div_info_dataset('','', '', '' , None)  ,{'textAlign': 'center', "visibility": "hidden"}






# Select columns and store in filtered-data-storage
@app.callback(  Output('preview_table','children'),
                Output('trainready_dataframe_storage','data'),
                #Output('continue_button_home','disabled'),
                Output('train_new_model_button','disabled'),
                Output('load_saved_model_button','disabled'),
                Input('processed_dataframe_storage','data'),
                Input('dropdown_col_df_names','value'),
                prevent_initial_call=True
               )
def update_table_preview(input_data,columns):

    if input_data is not None and len(columns)>0:
        dff = pd.read_json(input_data,orient='split')
        dff = dff[columns]
        disabled_button = False

    else:
        dff =   pd.DataFrame(columns=[])
        disabled_button = True


    data =  dff.to_json(date_format='iso',orient = 'split')
    table =  create_preview_table(dff)


    return table,data,disabled_button,disabled_button


# numerical_features_to_dropdown
@app.callback(  Output('dropdown_col_df_names','options'),
                Output('dropdown_col_df_names','value'),
                Input('processed_dataframe_storage','data'),
                prevent_initial_call=True
            )

def numerical_features_to_dropdown(input_data):

        if input_data is not None:
            dff = pd.read_json(input_data,orient='split')
            columns = dff.columns
        else:
            columns=[]


        options = []  # must be a list of dicts per option

        for n in columns:
            options.append({'label' : n, 'value': n})

        return options, columns
       




@app.callback(  Output("modal", "is_open"),
                Output('cabecera', 'is_open'),
                Output('collapse_modify_data_button', 'is_open'),
                Output('collapse_traing_sel_home','is_open' ),
                Input("trainready_dataframe_storage", "data"), 
                Input("close", "n_clicks"),
                State("modal", "is_open"),
                State("cabecera", "is_open"),
                State("collapse_modify_data_button", "is_open"),
                State('collapse_traing_sel_home','is_open' ),
                prevent_initial_call=True
)
def toggle_modal(input_data, n_clicks, modal_is_open, cabecera_is_open, button_is_open, traing_sel_home_isopen):
  
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if input_data is None:
        return False, True, False,False
    elif(n_clicks is None):
        return True, False, True, True
    elif(trigger_id =='close' ):
        return False, False, True, True
    elif(trigger_id == 'trainready_dataframe_storage' ):
        return modal_is_open, cabecera_is_open, button_is_open, traing_sel_home_isopen
    else:
        True, False, True ,True


                       


@app.callback(  Output("info_dataset_collapse", "is_open"),
                Input("modify_data_button", "n_clicks"),
                State("info_dataset_collapse", "is_open"),
                prevent_initial_call=True
)
#TODO
#@cache.memoize(timeout=60)  # in seconds
def toggle_collapse_info_dataset(n, is_open):
    if n:
        return not is_open
    return is_open




#Select train or load model
@app.callback(  Output('train_newmodel_collapse', 'is_open'),
                Output('loadmodel_collapse', 'is_open'),
                Output('train_new_model_button','outline'),
                Output('load_saved_model_button','outline'),
                Input('train_new_model_button','n_clicks'),
                Input('load_saved_model_button','n_clicks'),
                State('train_new_model_button','outline'),
                State('load_saved_model_button','outline'),
                State('train_newmodel_collapse','is_open'),
                State('loadmodel_collapse','is_open'),
                prevent_initial_call=True

)
def select_card_train_or_loadmodel_div(train_b, load_b,outline1,outline2, is_open1, is_open2):
    

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(button_id == 'train_new_model_button'):
        return not is_open1 ,False, not outline1, True
    else:
        return  False, not is_open2,True, not outline2


#Habilitar boton load_saved_model
@app.callback(  Output('load_model_buton','disabled'),
                Output('info_selected_model','is_open'),
                Output('info_selected_model','children'),
                Input('modelos_guardados_en_la_app_dropdown','value'),
                prevent_initial_call=True
)
def enable_load_saved_model_button(filename):
    if ( filename ):


        with open(DIR_SAVED_MODELS + filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            model_type= unserialized_data[0]
            columns_dtypes =  unserialized_data[1]

            col_names_badges = []

            for c in columns_dtypes.keys():
                col_names_badges.append( dbc.Badge(c, pill=True, color="info", className="mr-1"))
        
            children = html.Div(children= [


                html.Br(),
                html.Div(children = [
                    dbc.Badge('Model Vector Dimensionality:', pill=True, color="light", className="mr-1"),
                    dbc.Badge(len(columns_dtypes.keys()), pill=True, color="info", className="mr-1")
                    
                ]),
                html.Br(),


                dbc.Badge('Model Trained With Features:', pill=True, color="light", className="mr-1"),


                html.Div(children= col_names_badges, style={'margin': '0 auto','width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center','flex-wrap': 'wrap'}),
                html.Br(),
                html.Br()

            ])


                            
        

        
        return False, True, children
    else:
        return True,False, ''


#Boton de continuar
@app.callback(  Output('hidden_div_for_redirect_callback', 'children'),
                Output('alert_load_model','is_open'),
                Output('alert_load_model','children'),
                Input('load_model_buton', 'n_clicks'),
                Input('train_mode_som_button', 'n_clicks'),
                Input('train_mode_gsom_button', 'n_clicks'),
                Input('train_mode_ghsom_button', 'n_clicks'),
                State('trainready_dataframe_storage','data'),
                State('dataset_table_preview', 'selected_columns'),
                State('modelos_guardados_en_la_app_dropdown','value'),
                prevent_initial_call=True)
def analizar_datos_home( n_clicks_1,n_clicks_2,n_clicks_3,n_clicks_4, data, selected_col, filename ):


    selected_col_name = selected_col[0]
    session_data.set_target_name(selected_col_name)

    df = pd.read_json(data,orient='split')
    session_data.set_pd_dataframe(df)

    if(df[selected_col_name].dtype ==  np.float64):
        print('TARGET CONTINUOS')
        session_data.set_discrete_data(False)
    else:
        print('TARGET DISCRETOS')
        session_data.set_discrete_data(True)



    session_data.estandarizar_data()


    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(button_id == 'load_model_buton'):

        # Load data (deserialize)
        with open(DIR_SAVED_MODELS + filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            model_type= unserialized_data[0]
            columns_dtypes =  unserialized_data[1]
            model_info = unserialized_data[2]
            session_data.set_modelo(unserialized_data[3])

        if(len(session_data.get_colums_dtypes().keys()) != len(columns_dtypes.keys()) ):
            #dim no coincide
            return '', True, 'ERROR: Model dimensionality and selected Dataset ones are not the same. Please, edit the number of selected features before continue.'

        elif(session_data.get_colums_dtypes() != columns_dtypes ):
            #types no coinciden
            return '', True, 'ERROR: Model features-types and selected Dataset ones are not the same. Please, edit the selected features before continue.'

        else:
            #reorder selected dataframe cols to be the same as trained model
            cols = list(columns_dtypes.keys()) 
            df = session_data.get_pd_dataframe()[cols]
            session_data.set_pd_dataframe(df)

        if  model_type ==  'som':
            session_data.set_som_model_info_dict_direct(model_info)
            return dcc.Location(pathname=URLS['ANALYZE_SOM_URL'], id="redirect"), False, ''
        elif model_type ==   'gsom':
            session_data.set_gsom_model_info_dict_direct(model_info)
            return dcc.Location(pathname=URLS['ANALYZE_GSOM_URL'], id="redirect"), False, ''
        elif model_type ==   'ghsom':
            session_data.set_ghsom_model_info_dict_direct(model_info)
            return dcc.Location(pathname=URLS['ANALYZE_GHSOM_URL'], id="redirect"), False, ''
        else:   #if something goes worng 
            return dcc.Location(pathname="/", id="redirect"), False, ''


    elif (button_id == 'train_mode_som_button'):
        return dcc.Location(pathname=URLS['TRAINING_SOM_URL'], id="redirect"), False, ''
    elif(button_id == 'train_mode_gsom_button'):
        return dcc.Location(pathname=URLS['TRAINING_GSOM_URL'], id="redirect"), False, ''
    elif(button_id == 'train_mode_ghsom_button'):
        return dcc.Location(pathname=URLS['TRAINING_GHSOM_URL'], id="redirect"), False, ''
    else:   #if something goes wrong 
        return dcc.Location(pathname="/", id="redirect"), False, ''

