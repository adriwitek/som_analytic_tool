


def score_qe(estimator, X, y= None):
    return -estimator.som_model.quantization_error(X)

#dont use this one
def score_mqe(estimator, X, y= None):
    #print('calling mee',flush = True)
    _, mqe = estimator.som_model.get_qe_and_mqe_errors(X)
    return - mqe

def score_topographic_error(estimator, X, y= None):
    return -estimator.som_model.topographic_error(X)