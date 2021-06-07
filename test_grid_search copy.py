import models.sklearn_interface_for_som.sklearn_interface_for_som as sksom
from models.sklearn_interface_for_som.scoring_functions import score_mqe, score_topographic_error

from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import  datasets


#estimador = sksom.SOM_Sklearn()
estimador = sksom.SOM_Sklearn(  sigma=1.0, learning_rate=0.5,
                                neighborhood_function='gaussian', topology='rectangular',
                                num_iteration=None, random_seed=None)
#estimador.fit()
#check_estimator(estimador)

parameters = {  
                'hor_size': range(19,20),
                'ver_size' : range(19,20),
                'activation_distance': ('euclidean', 'cosine', 'manhattan', 'chebyshev'),
                'weights_init': ( 'PCA','random',None)
             }
        



scoring ={'qe': score_mqe,
           #'tp': score_topographic_error,
           'qe2': score_mqe,
        }


#scoring = score_qe

iris = datasets.load_iris()
gs = GridSearchCV(estimador, parameters, n_jobs = -1, scoring= scoring, refit = 'qe')
gs.fit(iris.data)
results = gs.cv_results_




print("RESULTS")
print('Best estimator', gs.best_estimator_ )
print('Best score', gs.best_score_ )
print('Best params', gs.best_params_ )
print('Best index', gs.best_index_ )
#if more than one metric ending is not _score, it is _paramname
print('Results[best index]_qe', results['mean_test_qe'][gs.best_index_])
#print('Results[best index]_tp', results['mean_test_tp'][gs.best_index_])
print('Results[best index]_qe2', results['mean_test_qe2'][gs.best_index_])

print()
print()
print()
print('---AHORA DE TODOSSSSSSSS')

print('Results[best index]_qe', results['mean_test_qe'])
#print('Results[best index]_tp', results['mean_test_tp'])
print('Results[best index]_qe2', results['mean_test_qe2'])

