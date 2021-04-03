#APP CONFIG



#Global Var.
APP_NAME = 'SOM Analytic Tool'
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

DEFAULT_HEATMAP_PX_HEIGHT = 700
DEFAULT_HEATMAP_PX_WIDTH = 700


#Models
GHSOM_MAX_SUBLEVELS_VIEW = 25
GHSOM_MAX_SUBMAPS_SIZE_VIEW = 200

#DIR
DIR_SAVED_MODELS='Saved_Data/'

DIR_APP_DATA= '.appdata/'
SESSION_DATA_FILE_DIR= DIR_APP_DATA + 'app_data.json'


#URLS
URLS= {
    'TRAINING_SELECTION_URL' : '/training_selection',
    'TRAINING_SOM_URL' : '/train-som', 
    'TRAINING_GSOM_URL' : '/train-gsom', 
    'TRAINING_GHSOM_URL' : '/train-ghsom', 
    'ANALYZE_SOM_URL' : '/analyze-som-data', 
    'ANALYZE_GSOM_URL' : '/analyze-gsom-data', 
    'ANALYZE_GHSOM_URL' : '/analyze-ghsom-data', 

}

