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
#import numpy as np

from  views.session_data import session_data
from  config.config import *
from  config.config import DEFAULT_DATASET_ROWS_PREVIEW
import views.plot_utils as pu


#import plotly.graph_objects as go

from os import listdir,makedirs
from os.path import isfile, join
import pickle
from math import ceil,floor
from pathlib import Path
import os





#TODO pasar a pu file
show_file_info_style =  {'textAlign': 'center',  'display': 'block'}
hidden_div_style ={'textAlign': 'center', "visibility": "hidden",'display':'none'} 




#############################################################
#	                       LAYOUT	                        #
#############################################################


#### TRAINING SELECTION
def Training_selection(): 

    layout = html.Div(id = 'training_selection_div',
        children=[


            #elements.navigation_bar,
            html.Div(   id='select_button', 
                        style=pu.get_css_style_inline_flex(),
                        children=[
                            dbc.Button("Train New Model", id="train_new_model_button", className="mr-2", color="primary", outline=True ),
                            dbc.Button("Load Pre-Trained Model", id="load_saved_model_button", className="mr-2", color="primary", outline=True  )
                        ]
            ),

            dbc.Collapse(id = 'train_newmodel_collapse',
                        is_open = False,
                        children = [
           
                        
                                dbc.Card(color = 'light',
                                    children=[
                                        dbc.CardHeader(html.H2('Train New Model')),

                                        dbc.CardBody(
                                            dbc.ListGroup([
                                            
                                                #SOM
                                                dbc.ListGroupItem([
                                                    html.H4('SOM',style=pu.get_css_style_center()  ),
                                                    html.Div( 
                                                        [dbc.Button("SOM", id="train_mode_som_button", className="mr-2", color="primary",)],
                                                        style=pu.get_css_style_center() 
                                                    )
                                                ]),


                                                #GSOM
                                                dbc.ListGroupItem([
                                                    html.H4('GSOM',style=pu.get_css_style_center()  ),
                                                    html.Div( 
                                                        [dbc.Button("GSOM", id="train_mode_gsom_button", className="mr-2", color="primary",)],
                                                        style=pu.get_css_style_center() 
                                                    )
                                                ]),


                                                #GHSOM
                                                dbc.ListGroupItem([
                                                    html.H4('GHSOM',style=pu.get_css_style_center()  ),
                                                    html.Div( 
                                                        [dbc.Button("GHSOM", id="train_mode_ghsom_button", className="mr-2", color="primary",)],
                                                        style=pu.get_css_style_center() 
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
    
                        dbc.Card(color = 'light',
                            children=[
                                dbc.CardHeader(html.H2('Load Pre-Trained Model')),
    
                                dbc.CardBody(
                                    dbc.ListGroup([
                                    
                                        # Modelos guardados en la app
                                        dbc.ListGroupItem([
                                            html.H4('Saved Models',className="card-title" , style=pu.get_css_style_center() ),
                                            html.Div(style=pu.get_css_style_center() ,
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
                                                        html.Div(id='hidden_div_for_load_model',style=pu.get_css_style_center()),
    
    
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

        #Dash componets for storing data(Only used for signaling, while dumping data i pickle to make it
        # quicker)
        dcc.Store(id='original_dataframe_storage',data=None),
        dcc.Store(id='processed_dataframe_storage',data=None),
        dcc.Store(id='notnumeric_dataframe_storage',data=None),
        # making quicker the home view this additional dcc store
        dcc.Store(id='head_processed_dataframe_storage',data=None),

        elements.navigation_bar,
        elements.cabecera,

        dbc.Card(color = 'light',children=[
            dbc.CardHeader(html.H2('Select Dataset')),

            dbc.CardBody(
                dbc.ListGroup([
                    # Archivo Local
                    dbc.ListGroupItem([
                        html.H4('Local File',className="card-title" , style=pu.get_css_style_center() ),

                        dcc.Upload( id='upload-data', 
                                    children= [ dcc.Loading(id='loading_file_animation',
                                                    type='dot',
                                                    children= get_upload_data_component_text()
                                                )    
                                    ],
                                    style={'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'},
                                    # Allow multiple files to be uploaded
                                    multiple=False
                        ),

                        dbc.Collapse(id ='collapse_error_uploadingfile',
                                    is_open = False,
                                    children = [   
                                        html.P( id= 'error_tag_uploading_file',
                                                children = 'ERROR: File format not admited' ,
                                                style=pu.get_css_style_center())
                                    ]
                        ),
                        html.Br(),
                        #File Info collapse
                        dbc.Collapse(   id ='collapse_correct_loaded_file',
                                        is_open = False,
                                        children = [
                                            dcc.Loading(id='loading',
                                                    type='dot',
                                                    children=[
                                                            dbc.Collapse(id ='collapse_div_info_loaded_file',
                                                                        is_open= False,
                                                                        children = [   
                                                                            html.Div(id='div_info_loaded_file',
                                                                                    style=hidden_div_style,
                                                                                    children = div_info_loaded_file('','','','')
                                                                            )
                                                                        ]
                                                            )
                                            ]),

                                            # Preview Table
                                            html.Div(id = 'preview_table' ,children = create_preview_table(None)),
                                            html.Br(),

                                            #info showed when the dataset its loaded
                                            dbc.Collapse(id='collapse_modify_data_button', is_open = False, children = 
                                                dbc.Button("Modify Data",id="modify_data_button",className="mb-6",color="primary",block=True)
                                            ),
                                            dbc.Collapse(id ='info_dataset_collapse',
                                                children = [   
                                                    html.Div(id='info_dataset_div',
                                                            style=hidden_div_style,
                                                            children = div_info_dataset( None,0 ) 
                                                    )
                                                ]
                                            ),

                                          
                                            html.Hr(),
                                            html.Br(),

                                            #Training Selection Card
                                            dbc.Collapse(id = 'collapse_traing_sel_home',
                                                is_open= False,
                                                children = [
                                                    html.Div(Training_selection())
                                                ]
                                            )
                                        ]
                        ),

                        
                    ]),
                ],flush=True,),
            )
        ]),



        



    ])

    return layout









#############################################################
#	                  AUX LAYOUT FUNS	                    #
#############################################################

#Used for spinner animation in uploading data
def get_upload_data_component_text():
    return html.Div(['Drag and Drop or  ', html.A('Click to Select File  (.csv)')])


def get_app_saved_models():

    makedirs(DIR_SAVED_MODELS, exist_ok=True)

    onlyfiles = [f for f in listdir(DIR_SAVED_MODELS) if isfile(join(DIR_SAVED_MODELS, f))]

    options = []  # must be a list of dicts per option
    for f in onlyfiles:
        if f.endswith('.pickle'):
            options.append({'label' : f, 'value': f})
    return options



 
def div_info_loaded_file(filename,fecha_modificacion, n_samples, n_features):
    return      html.Div(    id = 'div_info_loaded_file',
                            style=pu.get_css_style_inline_flex(),
                            children =[
                                html.H6( dbc.Badge( 'Filename:' ,  pill=True, color="light", className="mr-1")   ),
                                html.H6( dbc.Badge(filename, pill=True, color="info", className="mr-1")   ),
                                html.H6( dbc.Badge( 'with' ,  pill=True, color="light", className="mr-1")   ),

                                #html.H6( dbc.Badge(fecha_modificacion, pill=True, color="warning", className="mr-1")   ),
                                html.H6( dbc.Badge(str(n_samples) , id= 'badge_n_samples', pill=True, color="info", className="mr-1")   ),
                                html.H6( dbc.Badge( ' samples' , id= 'badge_n_samples', pill=True, color="light", className="mr-1")   ),

                                html.H6( dbc.Badge(str(n_features) , id= 'badge_n_features', pill=True, color="info", className="mr-1")   ),
                                html.H6( dbc.Badge( ' features' , id= 'badge_n_features', pill=True, color="light", className="mr-1")   )

                ])
    


# AUX FUN for #CARD DATASET SIZE EDIT
def get_split_menu(n_samples):

    children = [
        html.P('To let 100% of rows be train or test data, It\'s not necessary to split dataset',className="text-secondary",  style=pu.get_css_style_center()   ),
        #Train/Test percentage badge
        html.Div(style=pu.get_css_style_inline_flex(),
                children = [
                    html.H6(children='Train percentage  '),
                    html.H6( dbc.Badge( '50 %',  pill=True, color="warning", className="mr-1",id ='badge_info_percentage_train_slider')   ),
                    html.H6(children='Test percentage  '),
                    html.H6( dbc.Badge( '50 %',  pill=True, color="warning", className="mr-1",id ='badge_info_percentage_test_slider')   )
                ]
        ),

        #Slider split percentage
        dcc.Slider(id='split_slider', min=0,max=100,value=50,step =1 ,
                    marks={
                        0: {'label': '0 %'},
                        10: {'label': '10 %'},
                        25: {'label':  '25 %'},
                        50: {'label':  '50 %'},
                        75: {'label':  '75 %'},
                        100: {'label': '100 %'}
                    }
        ),

        #Number of samples train/test
        html.Div(style=pu.get_css_style_inline_flex(),
                children = [
                    html.H6(children='Train Samples\t'),
                    dcc.Input(id="train_samples_input", type="number", value=ceil(n_samples/2),step=1,min=0, max = n_samples),
                    html.H6(children='Test Samples\t'),
                    dcc.Input(id="test_samples_input", type="number", value=floor(n_samples/2),step=1,min=0, max = n_samples),
                ]
        ),
    ]

    return children



#CARD DATASET SIZE EDIT
def get_dataset_size_card(n_samples):

    return  dbc.Card(  color = 'light',
                        children=[
                            dbc.CardBody(children=[

                                html.Div(children=[

                                        #Dataset percentage
                                        html.Div(style=pu.get_css_style_inline_flex(),
                                                children = [
                                                    html.H6(children='Dataset percentage\t'),
                                                    html.H6( dbc.Badge( '100 %',  pill=True, color="warning", className="mr-1",id ='badge_info_percentagedataset_slider')   )
                                                ]
                                        ),

                                        #Slider percentage
                                        dcc.Slider(id='dataset_percentage_slider', min=ceil(100/n_samples),max=100,value=100,step =1 ,
                                                    marks={
                                                        0: {'label': '0 %'},
                                                        10: {'label': '10 %'},
                                                        25: {'label':  '25 %'},
                                                        50: {'label':  '50 %'},
                                                        75: {'label':  '75 %'},
                                                        100: {'label': '100 %'}
                                                    }
                                        ),

                                        #Number of samples
                                        html.Div(style=pu.get_css_style_inline_flex(),
                                                children = [
                                                    html.H6(children='Number of samples\t'),
                                                    dcc.Input(id="dataset_nsamples_input", type="number", value=n_samples,step=1,min=1, max = n_samples),
                                                ]

                                        ),

                                        #Split Dataset Train/Test
                                        dbc.Checklist(  options=[{"label": "Split Dataset in Train/Test", "value": 0}],
                                                                    value=[],
                                                                    id="check_split_dataset"
                                        ),

                                        dbc.Collapse(   id = 'collapse_split_dataset',
                                                        is_open= False,
                                                        children = [
                                                            dbc.Card( color = 'light',
                                                                    children=[
                                                                        dbc.CardBody(
                                                                            get_split_menu(n_samples)
                                                                        )
                                                            ])
                                                        ]
                                        ),
                                    ],
                                    style= pu.get_css_style_center() 
                                ),
                            ])
                ])



#CARD TARGET SELECTION
def get_select_target_card():
    return  dbc.Card(  color = 'light',
                        children=[
                                                                   
                dbc.CardBody(children=[

                            html.Div(children=[

                                    #Target
                                    html.H6('Target Selection:'),
                                    dcc.Dropdown(id='dropdown_target_selection',
                                               options=[],
                                               multi=False,
                                               value = []
                                    ),
                                ],
                                style=pu.get_css_style_center()
                            ),
                ])
            ])




#CARD FEATURES SELECTION
def get_features_selection_card():
     
    return  dbc.Card(  color = 'light',
                        children=[
                            dbc.CardBody(children=[

                                html.Div(children=[
                                        #Atrib names
                                        html.H6('Feature Selection:'),
                                        html.P('Select at least 2 features',className="text-secondary",  style=pu.get_css_style_center()  ),

                                        dbc.Checklist(
                                            options=[
                                                {"label": "Detect and Clean Categorical Data", "value": 1},
                                            ],
                                            value=[],
                                            id="clean_data_switch",
                                            switch=True,
                                        ),

                                        dcc.Dropdown(id='dropdown_feature_selection',
                                                   options=[],
                                                   multi=True,
                                                   value = []
                                        ),
                                    ],
                                    style=pu.get_css_style_center()
                                ),
                            ])  
             ])    



def div_info_dataset( df, n_samples):
    if(df is None):
        #Todo paar style a pu
        return html.Div(style = { "visibility": "hidden",'display':'none'}  ,
                #These elements are called in callbacks, show they need to exist at the load
                children = [
                #Necesario para la callback de entrenarcargar modelo
                #Target
                dcc.Dropdown(id='dropdown_target_selection',
                           options=[],
                           multi=False,
                           value = []
                ),
                html.Br(),           

                #Atrib names
                dcc.Dropdown(id='dropdown_feature_selection',
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
                


        ])



    return html.Div(id='output_uploaded_file',children=[

                dbc.Tabs(
                    id='tabs_edit_dataset_home',
                    active_tab='target_size_tab',
                    style =pu.get_css_style_inline_flex(),
                    children=[
                        dbc.Tab(get_dataset_size_card(n_samples), label = 'Dataset Size',tab_id='target_size_tab' ),
                        dbc.Tab(get_select_target_card() ,label = 'Target Selection',tab_id='target_select_tab'),
                        dbc.Tab( get_features_selection_card() ,label = 'Feature Selection',tab_id='feature_selection_card'),
                        dbc.Tab(get_onehot_childrendiv_menu() ,label = 'Apply One Hot Econding',tab_id='onehot_tab'),
                    ]
                ),


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
                      



    

#PREVIEW DATASET TABLE
def create_preview_table(df, selected_columns = []):

    if(df is None):
        return dash_table.DataTable(
                                            id='dataset_table_preview',
                                            data = [],
                                            column_selectable="single",
                                            columns=[],
                                        
                                            selected_columns=[])
        


    else:
      
        #if(isinstance(df.columns,pd.DatetimeIndex)):# df.columns DatetimeIndex object, error in dash_table
        #    df.columns = ['Feature_' + str(i) for i,_ in enumerate(df.columns)]
    
        return html.Div([

                    html.Br(),
                    html.H4('Table Preview',className="card-title" , style=pu.get_css_style_center()  ),
                    html.P('Select a column to mark it as Target',className="text-secondary",  style= pu.get_css_style_center() ),

                    dcc.Loading(id='loading',
                                type='dot',
                                children=[
                                    
                                    html.Div(style = {"overflow": "scroll"},
                                        children=
                                        dash_table.DataTable(
                                            id='dataset_table_preview',
                                            column_selectable="single",
                                            columns=[{"name": str(i), "id": str(i), "selectable": True} for i in df.columns],
                                            
                                            selected_columns=selected_columns,
                                            data=df.head(DEFAULT_DATASET_ROWS_PREVIEW).to_dict('records'),

                                            style_cell={'textAlign': 'center',
                                                        'textOverflow': 'ellipsis',
                                                        'overflowX': 'auto'
                                            },
 

                                            style_data_conditional=[

                                                #{
                                                # 'if': { 'state': 'selected'   },
                                                # 'background_color': '#D2F3FF'
                                                #},

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

        
 return  dbc.Card(  color = 'light',
                        children=[
                            dbc.CardBody(children=[ 
                                html.Div(
                                    children = [
                                        html.H5("Select Categorical Features to apply them One Hot:"),
                                        dcc.Dropdown(
                                            id='dropdown_features_toonehot',
                                            options=[],
                                            multi=True
                                        ),
                                        dbc.Checklist(  options=[{"label": "Consider empty values", "value": 0}],
                                                                            value=[],
                                                                            id="check_nanvalues_onehot"
                                        ),
                                        dbc.Button("Apply One Hot", id="apply_onehot_button",disabled = True, className="mr-2", color="primary")
                                    ]
                                )
                            ])
            ])
   




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
    # [] if no column selected
    dataset_table_preview =  [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

    return dataset_table_preview
   



@app.callback(  Output('dropdown_features_toonehot', 'options'),
                Input('notnumeric_dataframe_storage', 'data'),
                prevent_initial_call=True 
)
def onehot_dropdown_options(input_data):

    options = []  # must be a list of dicts per option
    if input_data is not None:
        #dff = pd.read_json(input_data,orient='split')
        with open(NOT_NUMERICAL_DF_PATH , 'rb') as handle:
            dff = pickle.load(handle)
        col = dff.columns
         
        for n in col:
            options.append({'label' : n, 'value': n})
    return options


#disable_apply_onehot_button
@app.callback(  Output('apply_onehot_button', 'disabled'),
                Input('dropdown_features_toonehot', 'value'),
                prevent_initial_call=True 
)
def disable_apply_onehot_button(names):
    if(not(names)):
        return True
    else:
        return False

#Process data
@app.callback(  Output('processed_dataframe_storage','data'),
                Output('notnumeric_dataframe_storage','data'),
                Output('dropdown_features_toonehot', 'value'),
                Input('original_dataframe_storage','data'),
                Input('clean_data_switch', 'value'),    
                Input('apply_onehot_button', 'n_clicks'),
                State('processed_dataframe_storage','data'),
                State('notnumeric_dataframe_storage','data'),
                State('dropdown_features_toonehot','value'),
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

            #notnum_df   = pd.read_json(notnum_data,orient='split')
            with open(NOT_NUMERICAL_DF_PATH , 'rb') as handle:
                notnum_df = pickle.load(handle)
            
            if( (processed_data is None) or (not names) or (notnum_df.empty ) ):
               #devolvemos tal cual pq no hay cambios
               return  processed_data,notnum_data, dash.no_update
            else:#applicamos onehot

                #processed_df = pd.read_json(processed_data,orient='split')
                with open(PROCESSED_DF_PATH , 'rb') as handle:
                    processed_df = pickle.load(handle)

                for n in names:
                    # use pd.pd.concat to join the new columns with your original dataframe
                    processed_df = pd.concat( [pd.get_dummies(notnum_df[n], prefix=str(n), dummy_na=check_nan), processed_df ],axis=1)
                    notnum_df.drop([n],axis=1, inplace=True)


                #processed_data = processed_df.to_json(date_format='iso',orient = 'split')
                with open(PROCESSED_DF_PATH, 'wb') as handle:
                    pickle.dump(processed_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #notnum_data = notnum_df.to_json(date_format='iso',orient = 'split')
                with open(NOT_NUMERICAL_DF_PATH, 'wb') as handle:
                    pickle.dump(notnum_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #return  processed_data,notnum_data, []
                return  'OK','OK', []
       

       #contexto del switch o del original_Dataframe: procesamos datos
        else:  
            
            #dff = pd.read_json(input_data,orient='split')
            with open(ORIGINAL_DF_PATH , 'rb') as handle:
                dff = pickle.load(handle)
           
            #if(isinstance(dff.columns,pd.DatetimeIndex)):# df.columns DatetimeIndex object, error in dash_table
            #    dff.columns = ['Feature_' + str(i) for i,_ in enumerate(dff.columns)]
            dff_num, dff_not_num = getNumericVars(dff,bool(clean_categorical_data))

            #processed_data = dff_num.to_json(date_format='iso',orient = 'split')
            with open(PROCESSED_DF_PATH, 'wb') as handle:
                pickle.dump(dff_num, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #notnum_data = dff_not_num.to_json(date_format='iso',orient = 'split')
            with open(NOT_NUMERICAL_DF_PATH, 'wb') as handle:
                pickle.dump(dff_not_num, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #return  processed_data,notnum_data, dash.no_update
            return  'OK','OK',  dash.no_update

    
    else:
        #dff = pd.DataFrame(columns=[])
        #return dff,dff, dash.no_update
        return None, None, dash.no_update









#Carga la info del dataset en el home
@app.callback(  Output('original_dataframe_storage', 'data'),
                Output('collapse_div_info_loaded_file','children'),
                Output('collapse_div_info_loaded_file','is_open'),
                Output('info_dataset_div', 'children'),
                Output('info_dataset_div', 'style'),

                Output('collapse_error_uploadingfile', 'is_open'),
                Output('collapse_correct_loaded_file', 'is_open'),
                Output('error_tag_uploading_file', 'children'),

                Output('loading_file_animation', 'children'),#Just for training spinner animation

                Input('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('upload-data', 'last_modified'), 
                prevent_initial_call=True
)
#TODO
#@cache.memoize(timeout=60)  # in seconds
def update_output( contents, filename, last_modified):
    '''Carga el dataset en los elementos adecuados

    '''
    show_file_info_style =  {'textAlign': 'center',  'display': 'block'}
    hidden_file_info_style = {'textAlign': 'center', "visibility": "hidden"}
    #TODO pasar estylos a pu

    if contents is not None:
        try:
            if 'csv' in filename:
                content_split = contents.split(',')
                if(len(content_split) <2):
                    return dash.no_update,'', False, '', hidden_file_info_style, True,False,'An error occurred processing the file', get_upload_data_component_text()
                content_type, content_string = content_split
                decoded = base64.b64decode(content_string)
                try:
                    # Assume that the user uploaded a CSV file
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = None, decimal = ",", engine='python')

                except: 
                    dataframe = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep = ',', decimal = "." , engine='python')


            else:
                return dash.no_update,'', False, '',hidden_file_info_style,True,False, 'ERROR: File format not admited', get_upload_data_component_text()

        except Exception as e:
            print(e)
            return dash.no_update,'', False, '', hidden_file_info_style, True,False,'An error occurred processing the file', get_upload_data_component_text()
        
    
        n_samples, n_features=dataframe.shape

        '''
        if(isinstance(dataframe.columns,pd.DatetimeIndex)):# df.columns DatetimeIndex object, error in dash_table
                dataframe.columns = ['Feature_' + str(i) for i,_ in enumerate(dataframe.columns)]
        print('Despues de ver si es datetimeindex----')
        print('dataframe.columns', dataframe.columns)
        '''


        if(n_samples == 0):
            return dash.no_update,'', False,'', hidden_file_info_style, True,False,'ERROR: The file does not contain any sample', get_upload_data_component_text()
        elif(n_features<=2):
            return dash.no_update,'', False, '', hidden_file_info_style, True,False,'ERROR: The file must contain at least 2 features', get_upload_data_component_text()
            
        div1 = div_info_loaded_file(filename,
                                    datetime.utcfromtimestamp(last_modified).strftime('%d/%m/%Y %H:%M:%S'),
                                    str(n_samples),
                                    str(n_features))
        div2 = div_info_dataset(dataframe,n_samples) 
        dirpath = Path(DIR_APP_DATA)
        if( not dirpath.exists()):
            os.mkdir(dirpath)
        with open(ORIGINAL_DF_PATH, 'wb') as handle:
            pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 'OK',div1, True,   div2, show_file_info_style , False,True, '', get_upload_data_component_text()
        #return dataframe.to_json(date_format='iso',orient = 'split'),div1, True,   div2, show_file_info_style , False,True, ''

                
    else: 
        return  None,'' , False,  div_info_dataset( None,0) ,hidden_file_info_style, False,True, '', get_upload_data_component_text()






# numerical_features_to_dropdown
@app.callback(  Output('head_processed_dataframe_storage','data'),
                Output('dropdown_feature_selection','options'),
                Output('dropdown_target_selection','options'),
                Input('processed_dataframe_storage','data'),
                State('notnumeric_dataframe_storage','data'),
            )

def processed_df_callback(processed_df,not_numeric_df):

    if processed_df is not None:
        #dff1 = pd.read_json(processed_df,orient='split')
        with open(PROCESSED_DF_PATH , 'rb') as handle:
            dff1 = pickle.load(handle)
        processed_cols = dff1.columns.tolist()
        #print('processed_cols', processed_cols)
    else:
        raise PreventUpdate
  
    if(not_numeric_df is not None):
        #dff2 = pd.read_json(not_numeric_df,orient='split')
        with open(NOT_NUMERICAL_DF_PATH , 'rb') as handle:
            dff2 = pickle.load(handle)
        notnum_cols = dff2.columns.tolist() 
 
    #Features options = processed df
    options_feature_selection = []  # must be a list of dicts per option
    for n in processed_cols:
        options_feature_selection.append({'label' : n, 'value': n})

    #Targets options = processed df + notnum df
    options_target_selection = options_feature_selection.copy() 
    for n in notnum_cols:
        options_target_selection.append({'label' : n, 'value': n})

    #For preview table
    #head = dff1.head(DEFAULT_DATASET_ROWS_PREVIEW).to_json(date_format='iso',orient = 'split') 
    head = dff1.head(DEFAULT_DATASET_ROWS_PREVIEW)
    #print('saving head',type(head))
    with open(HEAD_PROCESSED_DF_PATH, 'wb') as handle:
        pickle.dump(head, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if(head is None):
        return  None, options_feature_selection, options_target_selection
    else:
        return  'OK', options_feature_selection, options_target_selection


       


@app.callback(  Output('dropdown_feature_selection','value'),
                Output('dropdown_target_selection','value'),
                Input('dropdown_feature_selection','value'),
                Input('dropdown_target_selection','value'),
                Input('dropdown_feature_selection','options'),
                Input('dataset_table_preview', 'selected_columns'),
                prevent_initial_call=True
)
def sync_target_and_feature_selection(features_values,target_value, feature_options,table_selected_col):

    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]

    if(trigger == 'dropdown_feature_selection.options'): #first call select all
        all_features_values = []
        for dic in feature_options:
            all_features_values.append(dic['label'])
        return all_features_values, None

    else:#rest of calls
        if(trigger =='dataset_table_preview.selected_columns' and table_selected_col is not None and len(table_selected_col)>0 ):
            target_value = table_selected_col[0]

        if(target_value is None or len(target_value)==0): #no target
            return dash.no_update, dash.no_update
        else:
            if( target_value in features_values):
                features_values.remove(target_value)
                return features_values, target_value
            else:
                return dash.no_update, target_value

    



@app.callback(  Output('preview_table','children'),
                Output('train_new_model_button','disabled'),
                Output('load_saved_model_button','disabled'),
                Input('head_processed_dataframe_storage','data'),
                Input('dropdown_feature_selection','value'),
                Input('dropdown_target_selection','value'),
                State('dataset_nsamples_input','value'),
                State('notnumeric_dataframe_storage', 'data'),
                prevent_initial_call=True
)
def callback_preview_table(input_data,features_values,target_value,n_of_samples,notnumeric_df):

    disabled_button = True
    selected_columns = []
    
    if (input_data is  None) or  ( len(features_values)==0 and (target_value is  None or (len(target_value) ==0 )) ):
        dff =   pd.DataFrame(columns=[])
    else:#there are features and/or target to show
        #df = pd.read_json(input_data,orient='split')
        with open(HEAD_PROCESSED_DF_PATH , 'rb') as handle:
            df = pickle.load(handle)

        if(len(features_values)==1 ):
            disabled_button = True
            dff = df[features_values]
        elif(len(features_values)> 1):
            disabled_button = False
            dff = df[features_values]
        else:  
            dff =   pd.DataFrame(columns=[])

        #Add target col
        if( target_value is not None and (len(target_value) >0 )):
            selected_columns = [target_value]
            if(target_value not in df.columns):
                #notnumeric_df = pd.read_json(notnumeric_df,orient='split')
                with open(NOT_NUMERICAL_DF_PATH , 'rb') as handle:
                    notnumeric_df = pickle.load(handle)
                dff = pd.concat( [dff, notnumeric_df[target_value] ],axis=1)
            else:
                dff = pd.concat( [dff, df[target_value] ],axis=1)

    if(n_of_samples < DEFAULT_DATASET_ROWS_PREVIEW):#if dataset is very small
        dff = dff.head(n_of_samples)
        if(n_of_samples == 0):
            disabled_button = True

    table =  create_preview_table(dff,selected_columns = selected_columns )

    return table, disabled_button, disabled_button




@app.callback(  Output("modal", "is_open"),
                Output('cabecera', 'is_open'),
                Output('collapse_modify_data_button', 'is_open'),
                Output('collapse_traing_sel_home','is_open' ),
                Input("head_processed_dataframe_storage", "data"), 
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
        return True, False, True ,True


                       


@app.callback(  Output("info_dataset_collapse", "is_open"),
                Input("modify_data_button", "n_clicks"),
                State("info_dataset_collapse", "is_open"),
                prevent_initial_call=True
)
def toggle_collapse_info_dataset(n, is_open ):
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
                Input('dropdown_feature_selection','value'),
                State('train_new_model_button','outline'),
                State('load_saved_model_button','outline'),
                State('train_newmodel_collapse','is_open'),
                State('loadmodel_collapse','is_open'),
                prevent_initial_call=True

)
def select_card_train_or_loadmodel_div(train_b, load_b,features_values, outline1,outline2, is_open1, is_open2):
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(  len(features_values)==0 ):
        return False,False,True,True
    if(button_id == 'train_new_model_button'):
        return not is_open1 ,False, not outline1, True
    elif(button_id == 'load_saved_model_button'):
        return  False, not is_open2,True, not outline2
    else:
        return False,False,True,True




#Habilitar boton load_saved_model
@app.callback(  Output('load_model_buton','disabled'),
                Output('info_selected_model','is_open'),
                Output('info_selected_model','children'),
                Input('modelos_guardados_en_la_app_dropdown','value'),

                Input('dataset_nsamples_input', 'value'),
                Input('check_split_dataset', 'value'),
                Input('train_samples_input', 'value'),
                Input('test_samples_input', 'value'),
                prevent_initial_call=True
)
def enable_load_saved_model_button(filename, n_samples,check, n_train_samples , n_test_samples):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if(trigger_id != 'modelos_guardados_en_la_app_dropdown'  ):#Disable button if no correct data number of samples selected
        if(n_samples is None or n_samples <= 0 ):
            return True, dash.no_update, dash.no_update
        elif(check and (n_test_samples is None or n_test_samples <= 0 or 
                        n_train_samples is None or n_train_samples <0)):
            return True, dash.no_update, dash.no_update
        else:
            if ( filename ):
                return False,dash.no_update, dash.no_update
            else:
                return True, dash.no_update, dash.no_update


    else: # modelos_guardados_en_la_app_dropdown
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
                    html.Div(children= col_names_badges, style=pu.get_css_style_inline_flex()),
                    html.Br(),
                    html.Br()
                ])

            return False, True, children
        else:
            return True,False, ''


# Sync slider 
@app.callback(  Output("dataset_percentage_slider", "value"),
                Output("badge_info_percentagedataset_slider", "children"),
                Output("dataset_nsamples_input", "value"),
                Input("dataset_percentage_slider", "value"),
                Input("dataset_nsamples_input", "value"),
                State('dataset_nsamples_input', 'max'),
                prevent_initial_call=True
                )
#todo en vez de leer json almacenar en un store
def sync_slider_input_datasetpercentage(slider_percentage, n_samples_sel, n_samples):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if (trigger_id == "dataset_percentage_slider"):
        if(slider_percentage is None):
            raise dash.exceptions.PreventUpdate
        else:
            number = (slider_percentage * n_samples)/100
            number = ceil(number)
            percentage = (number*100)/n_samples
            return percentage, (str(percentage) + ' %'), number
    else:
        if(n_samples_sel is None or n_samples_sel >n_samples ):
            raise dash.exceptions.PreventUpdate
        else:
            percentage = (n_samples_sel*100)/n_samples
            return percentage, (str(  round(percentage, 2 )) + ' %'),dash.no_update




@app.callback(
            Output('collapse_split_dataset', 'is_open'),
            Output('check_split_dataset', 'value'),
            Input('check_split_dataset', 'value'),
            Input("dataset_nsamples_input", "value"),
            prevent_initial_call=True
    
)
def split_dataset_collapse(check, n_sel_samples):

    if(check):
        if(n_sel_samples is None or n_sel_samples< 2):
            return False, 0
        else:
            return True, dash.no_update
    else:
        return False, dash.no_update


# Sync slider split train test
@app.callback(  Output("split_slider", "value"),
                Output("badge_info_percentage_train_slider", "children"),
                Output("badge_info_percentage_test_slider", "children"),
                Output("train_samples_input", "value"),
                Output("test_samples_input", "value"),
                Input("split_slider", "value"),
                Input("train_samples_input", "value"),
                Input("test_samples_input", "value"),
                Input("dataset_nsamples_input", "value"),
                prevent_initial_call=True
                )
#todo en vez de leer json almacenar en un store
def sync_slider_split(slider_percentage, train_samples_sel,test_samples_sel, n_of_sel_samples): 
   
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(n_of_sel_samples is None or n_of_sel_samples < 1):
        raise dash.exceptions.PreventUpdate

    elif(trigger_id == 'dataset_nsamples_input' ):
        return 50, (str(50) + ' %'), (str(50) + ' %'), ceil(n_of_sel_samples/2), floor(n_of_sel_samples/2), 

    elif (trigger_id == "split_slider"):
        if(slider_percentage is None):
            raise dash.exceptions.PreventUpdate
        else:
            number = (slider_percentage * n_of_sel_samples)/100
            number = ceil(number)
            #percentage = (number*100)/n_of_sel_samples
            percentage = round ((number*100)/n_of_sel_samples, 2)

            return percentage, (str(percentage) + ' %'), (str((100 - percentage)) + ' %'), number, (n_of_sel_samples - number)

    elif(train_samples_sel is None or test_samples_sel is None ):
        raise dash.exceptions.PreventUpdate

    elif(trigger_id == 'train_samples_input'):
        percentage_train = (train_samples_sel*100)/n_of_sel_samples
        percentage_test =  100 - percentage_train
        test_samples_sel =n_of_sel_samples - train_samples_sel
        return percentage_train, (str(  round(percentage_train, 2 )) + ' %'),(str(  round(percentage_test, 2 )) + ' %'),train_samples_sel,    test_samples_sel

    else:
        percentage_test = (test_samples_sel*100)/n_of_sel_samples
        percentage_train =  100 - percentage_test
        train_samples_sel =n_of_sel_samples - test_samples_sel
        return percentage_train, (str(  round(percentage_train, 2 )) + ' %'),(str(  round(percentage_test, 2 )) + ' %'),train_samples_sel,    test_samples_sel



#disable train the 3 models train buttons if split is check and train numebr is 0
@app.callback(  Output('train_mode_som_button','disabled'),
                Output('train_mode_gsom_button','disabled'),
                Output('train_mode_ghsom_button','disabled'),
                Input('dataset_nsamples_input', 'value'),
                Input('check_split_dataset', 'value'),
                Input('train_samples_input', 'value'),
                Input('test_samples_input', 'value'),
                prevent_initial_call=True
)
def enable_train_models_buttons(n_samples,check, n_train_samples , n_test_samples):
    #print('n_samples', n_samples)
    #print('check', check)

    if(n_samples is None or n_samples <= 0 ):
        return True, True, True
    elif(check and (n_train_samples is None or n_train_samples <= 0 or 
                    n_test_samples is None or n_test_samples <0)):
        return True, True, True
    else:
        return False, False, False


#Boton de continuar
@app.callback(  Output('hidden_div_for_redirect_callback', 'children'),
                Output('alert_load_model','is_open'),
                Output('alert_load_model','children'),
                Input('load_model_buton', 'n_clicks'),
                Input('train_mode_som_button', 'n_clicks'),
                Input('train_mode_gsom_button', 'n_clicks'),
                Input('train_mode_ghsom_button', 'n_clicks'),
                State('processed_dataframe_storage','data'),
                State('notnumeric_dataframe_storage','data'),
                State('modelos_guardados_en_la_app_dropdown','value'),
                State("dataset_percentage_slider", "value"),
                State('dataset_nsamples_input','value'),
                State('dropdown_target_selection','value'),
                State('dropdown_feature_selection','value'),
                State('check_split_dataset','value'),
                State("train_samples_input", "value"),
                prevent_initial_call=True
)
def analizar_datos_home( n_clicks_1,n_clicks_2,n_clicks_3,n_clicks_4, data, notnumeric_df, filename,nsamples_percentage, nsamples_selected,
                         target_selection, feature_selection,check_split_dataset,train_samples_input  ):

    #df = pd.read_json(data,orient='split')
    with open(PROCESSED_DF_PATH , 'rb') as handle:
        df = pickle.load(handle)
    #notnumeric_df = pd.read_json(notnumeric_df,orient='split')
    with open(NOT_NUMERICAL_DF_PATH , 'rb') as handle:
        notnumeric_df = pickle.load(handle)
    df_features = df[feature_selection]
    df_targets =  pd.concat( [notnumeric_df,  df[df.columns.difference(feature_selection)] ],axis=1)  

    print('\t -->Shuffling Data...')

    if(nsamples_percentage != 100):

        df_features = df_features.sample(n=nsamples_selected, replace=False)
        df_targets = df_targets.loc[df_features.index,:]

        '''
        print('DEBUG:sampling random selection:')
        print('nsamples_selected',nsamples_selected)
        print('train_samples_input',train_samples_input)
        print('df_features',df_features)
        print('df_targets',df_targets)
        '''
    else:
        df_features = df_features.sample(frac=1, replace=False)
        df_targets = df_targets.loc[df_features.index,:]

    print('\t -->Shuffling Complete.')

    #preselected target
    if(target_selection is not None):
        session_data.set_target_name(str(target_selection))


    if(check_split_dataset and (train_samples_input == 0 or train_samples_input == nsamples_selected)):
        split = False
    else:
        split = check_split_dataset

    session_data.set_pd_dataframes(df_features,df_targets, split = split, train_samples_number=train_samples_input )

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if(button_id == 'load_model_buton'):

        # Load data (deserialize)
        with open(DIR_SAVED_MODELS + filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            model_type= unserialized_data[0]
            columns_dtypes =  unserialized_data[1]
            model_info = unserialized_data[2]
            session_data.set_modelos(unserialized_data[3])

        if(len(session_data.get_features_dtypes().keys()) != len(columns_dtypes.keys()) ):
            #dim no coincide
            return '', True, 'ERROR: Model dimensionality and selected Dataset ones are not the same. Please, edit the number of selected features before continue.'

        elif(session_data.get_features_dtypes() != columns_dtypes ):
            #types no coinciden
            return '', True, 'ERROR: Model features-types and selected Dataset ones are not the same. Please, edit the selected features before continue.'

        else:
            #reorder selected dataframe cols to be the same as trained model
            cols = list(columns_dtypes.keys()) 
            df_features = df_features[cols]
            session_data.set_pd_dataframes(df_features,df_targets,split = split, train_samples_number=train_samples_input )

        if  model_type ==  'som':
            session_data.set_som_model_info_dict_direct(model_info)
            session_data.convert_test_data_tonumpy()
            return dcc.Location(pathname=URLS['ANALYZE_SOM_URL'], id="redirect"), False, ''
        elif model_type ==   'gsom':
            session_data.set_gsom_model_info_dict_direct(model_info)
            session_data.convert_test_data_tonumpy()
            return dcc.Location(pathname=URLS['ANALYZE_GSOM_URL'], id="redirect"), False, ''
        elif model_type ==   'ghsom':
            session_data.set_ghsom_model_info_dict_direct(model_info)
            session_data.convert_test_data_tonumpy()
            return dcc.Location(pathname=URLS['ANALYZE_GHSOM_URL'], id="redirect"), False, ''
        else:   #if something goes worng 
            return dcc.Location(pathname="/", id="redirect"), False, ''

    elif (button_id == 'train_mode_som_button'):
        #session_data.convert_train_data_tonumpy()
        return dcc.Location(pathname=URLS['TRAINING_SOM_URL'], id="redirect"), False, ''
    elif(button_id == 'train_mode_gsom_button'):
        #session_data.convert_train_data_tonumpy()
        return dcc.Location(pathname=URLS['TRAINING_GSOM_URL'], id="redirect"), False, ''
    elif(button_id == 'train_mode_ghsom_button'):
        #session_data.convert_train_data_tonumpy()
        return dcc.Location(pathname=URLS['TRAINING_GHSOM_URL'], id="redirect"), False, ''
    else:   #if something goes wrong, but app never should get here anyway
        return dcc.Location(pathname="/", id="redirect"), False, ''




