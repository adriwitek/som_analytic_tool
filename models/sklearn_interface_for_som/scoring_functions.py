


def score_mqe(estimator, X, y= None):
    return -estimator.som_model.quantization_error(X)

def score_qe(estimator, X, y= None):
    qe, _ = -estimator.get_qe_and_mqe_errors(X)
    return - qe

def score_topographic_error(estimator, X, y= None):
    return -estimator.som_model.topographic_error(X)