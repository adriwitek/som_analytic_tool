import models.sklearn_interface_for_som.sklearn_interface_for_som as sksom
from models.sklearn_interface_for_som.scoring_functions import score_qe, score_topographic_error

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
        



scoring ={'qe': score_qe,
           'tp': score_topographic_error}


#scoring = score_qe

iris = datasets.load_iris()
clf = GridSearchCV(estimador, parameters, scoring= scoring)
#clf = GridSearchCV(estimador, parameters, n_jobs = -1, scoring= score_topographic_error)

clf.fit(iris.data)




print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


print("Best parameters set found on development set:")
print()
print(clf.best_params_)