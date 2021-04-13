#APP CONFIG



#Global Var.
APP_NAME = 'SOM Analytic Tool'
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"


#HEATMAPS CONF.
DEFAULT_HEATMAP_PX_HEIGHT = 500
DEFAULT_HEATMAP_PX_WIDTH = 500
DEFAULT_HEATMAP_COLORSCALE = 'Viridis'
#DEFAULT_HEATMAP_COLORSCALE = 'Jet'



''' COLORSCALES
aggrnyl     agsunset    blackbody   bluered     blues       blugrn      bluyl       brwnyl
bugn        bupu        burg        burgyl      cividis     darkmint    electric    emrld
gnbu        greens      greys       hot         inferno     jet         magenta     magma
mint        orrd        oranges     oryel       peach       pinkyl      plasma      plotly3
pubu        pubugn      purd        purp        purples     purpor      rainbow     rdbu
rdpu        redor       reds        sunset      sunsetdark  teal        tealgrn     turbo
viridis     ylgn        ylgnbu      ylorbr      ylorrd      algae       amp         deep
dense       gray        haline      ice         matter      solar       speed       tempo
thermal     turbid      armyrose    brbg        earth       fall        geyser      prgn
piyg        picnic      portland    puor        rdgy        rdylbu      rdylgn      spectral
tealrose    temps       tropic      balance     curl        delta       oxy         edge
hsv         icefire     phase       twilight    mrybm       mygbm
'''

#Models
GHSOM_MAX_SUBLEVELS_VIEW = 25
GHSOM_MAX_SUBMAPS_SIZE_VIEW = 200

#DIR
DIR_SAVED_MODELS='Saved_Data/'




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

