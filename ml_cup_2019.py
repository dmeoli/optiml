import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from utils import load_ml_cup, mean_euclidean_error, load_ml_cup_blind, plot_validation_curve, plot_learning_curve

if __name__ == '__main__':
    X, y = load_ml_cup()

    gamma_range = np.logspace(-6, -1, 5)
    C_range = [1, 10, 100, 200, 300, 400, 500, 600, 1000]
    epsilon_range = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

    from sklearn.metrics.pairwise import laplacian_kernel

    tuned_parameters = {'estimator__kernel': ['rbf'],
                        'estimator__epsilon': epsilon_range,
                        'estimator__C': C_range,
                        'estimator__gamma': gamma_range}

    from sklearn.svm import SVR as SKLSVR

    grid = GridSearchCV(MultiOutputRegressor(SKLSVR()), param_grid=tuned_parameters,
                        scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
                        cv=5,  # 5 fold cross validation
                        n_jobs=-1,  # use all processors
                        refit=True,  # refit the best model on the full dataset
                        verbose=True)
    grid.fit(X, y)

    print(f'best parameters: {grid.best_params_}')
    print(f'best score: {-grid.best_score_}')

    scorer = make_scorer(mean_euclidean_error)

    # plot validation curve to visualize the performance metric over a
    # range of values for some hyperparameters (C, gamma, epsilon, etc.)
    for (param_name, param_range) in tuned_parameters.items():
        if 'kernel' not in param_name:
            plot_validation_curve(grid.best_estimator_, X, y, param_name, param_range, scorer)

    # plot learning curve to visualize the effect of the
    # number of observations on the performance metric
    plot_learning_curve(grid.best_estimator_, X, y, scorer, train_sizes=np.linspace(.1, 1.0, 5))

    # save predictions on the blind test set
    np.savetxt('./ml/data/ML-CUP19/dmeoli_ML-CUP19-TS.csv', grid.predict(load_ml_cup_blind()), delimiter=',')
