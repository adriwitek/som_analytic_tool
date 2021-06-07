"""
Interface to adapt SOM algorithm(Minisom) with sklearn, in order to use hyper-parameters tuning
@author: Adrián Rubio
Created: 06-05-2021
"""

from sklearn.utils.estimator_checks import check_estimator
from models.som import minisom
from numpy.linalg import norm
from sklearn.base import BaseEstimator



class SOM_Sklearn():
    """
        Interface to adapt SOM algorithm(Minisom) with sklearn, in order to use hyper-parameters tuning.
        Due to sklearn reqs. this class wont have heritage from minisom
    """

    def __init__(self, ver_size = 3,hor_size = 3,sigma=1.0, learning_rate=0.5,
                 neighborhood_function='gaussian', topology='rectangular',
                 num_iteration=None, random_seed=None,
                 square_grid_size=None, activation_distance='euclidean',weights_init='PCA' ):
        
        '''Initializes a Minisom sklearn Interface

        Hyper-parameters tuning only will estimate x,y (som size), num_iteration(max num of iterations),
         activation fun(activation_distance fun.) and weights initialization

        Parameters
        ----------

        ver_size : int
            x dimension of the SOM.(Will be ignored if fit fun it's called with square_grid_size != None)

        hor_size : int
            y dimension of the SOM.(Will be ignored if fit fun it's called with square_grid_size != None)

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'


        random_seed : int, optional (default=None)
            Random seed to use.
        
        '''
        self.ver_size = ver_size
        self.hor_size = hor_size
        self.sigma=sigma 
        self.learning_rate=learning_rate
        self.neighborhood_function=neighborhood_function
        self.topology=topology
        self.random_seed=random_seed
        self.num_iteration=num_iteration
        self.square_grid_size=square_grid_size
        self.activation_distance=activation_distance
        self.weights_init=weights_init
       


    #session_data.set_show_error_evolution(False)#poner esto antes de llamar a fit....######################################################################

    def fit(self, X,y=None ): #Only last fit will save model!
        '''
        Trains a som model due to fit parameters
        Parameters
        ----------
        X : np.array or list
            Data matrix.
        hor_size : int
            x dimension of the SOM.
        ver_size : int
            y dimension of the SOM.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        activation_distance : string, optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'
        weights_init: string
            PCA,random or None for initial map weights initialization
        '''

        #print('X data dypte is',X.dtype,flush=True)
        self.input_len=X.shape[1]
        

        if(self.square_grid_size is None):
            self.x_grid_size = self.ver_size
            self.y_grid_size = self.hor_size
        else:
            self.x_grid_size = self.square_grid_size
            self.y_grid_size = self.square_grid_size
           
        if(self.num_iteration is None or self.num_iteration <1):
            self.num_iteration = X.shape[1]

        #x=hor_size, y=ver_size,
        som = minisom.MiniSom(x=self.x_grid_size, y=self.y_grid_size, input_len=X.shape[1], sigma=self.sigma, learning_rate=self.learning_rate,
                                neighborhood_function=self.neighborhood_function, topology=self.topology,
                                activation_distance=self.activation_distance, random_seed=self.random_seed)
        
        #Weigh init
        if(self.weights_init == 'pca'):
            som.pca_weights_init(X)
        elif(self.weights_init == 'random'):   
            som.random_weights_init(X)
        som.train(X, self.num_iteration, random_order=False, verbose=True) 
        self.som_model = som
        print('Size',self.square_grid_size,'MQE',som.get_qe_and_mqe_errors(X)[1],'dsitance', self.activation_distance ,
                'weight init', self.weights_init,  flush = True)
        return self



    #Scoring fun (QE fun)
    def score(self, X, y=None):
        """Returns the quantization error(negated for score fun) computed as the average
        distance between each input sample and its best matching unit."""
        return  - self.som_model.quantization_error(X)

    #Scoring fun (QE fun)
    def score_quantization_error(self, X, y =None):
        """Returns the quantization error(negated for score fun) computed as the average
        distance between each input sample and its best matching unit."""
        return  - self.som_model.quantization_error(X)

    def score_topographic_error(self, X, y=None):
        """Returns the topographic error (negated for score fun) computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        return  - self.som_model.topographic_error(X)

    def _more_tags(self):
        return {'allow_nan ': True}


    def get_params(self, deep=False):
        return {    
                    "ver_size": self.ver_size,
                    "hor_size": self.hor_size,
                    "sigma": self.sigma,
                    "learning_rate": self.learning_rate,
                    "neighborhood_function": self.neighborhood_function ,
                    "topology": self.topology,
                    "random_seed": self.random_seed,
                    "num_iteration": self.num_iteration,
                    "square_grid_size":self.square_grid_size ,
                    "activation_distance": self.activation_distance,
                    "weights_init":self.weights_init
                }

    def set_params(self, **parameters):
        '''
            The input is a dict of the form 'parameter': value
            These params should be the ones called in the __init__ method
        '''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




