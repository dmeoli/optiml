if __name__ == '__main__':
    from optiml.ml.svm import SVC
    from optiml.ml.utils import generate_linearly_separable_data, plot_svm_hyperplane

    from optiml.ml.svm.kernels import linear
    from optiml.optimization.unconstrained.stochastic import StochasticGradientDescent

    from sklearn.svm import SVC as SKLSVC
    from sklearn.model_selection import train_test_split

    X, y = generate_linearly_separable_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    svc = SVC(kernel=linear, optimizer=StochasticGradientDescent).fit(X, y)
    print(f'custom svc accuracy score is {svc.score(X_test, y_test)}')
    print(f'custom svc found {len(svc.support_)} support vectors from {len(X_train)} examples')
    print(f'custom svc w is {svc.coef_}')
    print(f'custom svc b is {svc.intercept_}')
    plot_svm_hyperplane(svc, X_train, y_train)

    svc = SKLSVC(kernel='linear').fit(X_train, y_train)
    print(f'sklearn svc accuracy score is {svc.score(X_test, y_test)}')
    print(f'sklearn svc found {len(svc.support_)} support vectors from {len(X_train)} examples')
    print(f'sklearn svc w is {svc.coef_}')
    print(f'sklearn svc b is {svc.intercept_}')
    plot_svm_hyperplane(svc, X_train, y_train)

    # TODO
    #  - nel file delle svm dopo il primo caso sostituire il dot product con K(xi xj)
    #  - modificare abstract del report con il nuovo nome e la nuova motivazione
    #  - aggiungere nel file support vector machines una parte con le primal solution tramite SGD o
    #    Subgradient ed eliminare i paring con sklear alla fine, sono inappropriati
    #  - aggiornare readme ed aggiungere contribute con shrinking heuristic, weighted SVMs ed inplace
    #    kernel computation with cache for SVMs
    #  - nel file stochastic optimizers è giusto mostrare loss ed r2 (?)
    #  - nel caso della lagrangiana modificare cost in dcost ed aggiungere pcost con la primal solution
    #  -
    #  - aggiungere gli altri schedulers al file (time-based decay, step decay ed exponential decay)
    #  - aggiungere LBFGS e conjugate gradient for linear functions da climin
    #  - formule per il conjugate gradient (mergiare fletcher reeves e polak ribiere?) come in climin?
    #  - provare a far funzionare il metodo di newton con le loss functions tramite autograd per calcolare l'hessiano
    #  - prendere metodi e doc da climin più possibile ed aggiungerla ai jupyter con pseudocodice
    #    anche dal pdf delle slides e cercare di costruire una documentazione con build come Alessandro
    #  -
    #  - capire perchè l'algoritmo SMO custom ha qualche problema nella grid search rispetto a sklean
    #    (problema di normalizzazione o scaler o di euristica shrinking?)
    #  -
    #  - fare dei test di NNClassifier per i multi-label
    #  - fixare il channel_last=False con gli initializers

    # from sklearn.metrics import make_scorer2
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.multioutput import MultiOutputRegressor
    #
    # from optiml.ml.utils import load_ml_cup, mean_euclidean_error, load_ml_cup_blind, plot_validation_curve, \
    #     plot_learning_curve
    #
    # X, y = load_ml_cup()
    #
    # gamma_range = [1e-8, 1e-6, 1e-4, 1e-2, 1]
    # C_range = [0.1, 1, 10, 100, 1000, 1500, 2000, 2500]
    # epsilon_range = [0.0001, 0.001, 0.1, 0.2, 0.3]
    #
    # from sklearn.metrics.pairwise import laplacian_kernel
    #
    # tuned_parameters = {'estimator__kernel': ['rbf', laplacian_kernel],
    #                     'estimator__epsilon': epsilon_range,
    #                     'estimator__C': C_range,
    #                     'estimator__gamma': gamma_range}
    #
    # from sklearn.svm import SVR as SKLSVR
    #
    # grid = GridSearchCV(MultiOutputRegressor(SKLSVR()), param_grid=tuned_parameters,
    #                     scoring=make_scorer(mean_euclidean_error, greater_is_better=False),
    #                     cv=5,  # 5 fold cross validation
    #                     n_jobs=-1,  # use all processors
    #                     refit=True,  # refit the best model on the full dataset
    #                     verbose=True)
    # grid.fit(X, y)
    #
    # df = DataFrame(grid.cv_results_)
    # print(df)
    #
    # print(f'best parameters: {grid.best_params_}')
    # print(f'best score: {-grid.best_score_}')
    #
    # scorer = make_scorer(mean_euclidean_error)
    #
    # # plot validation curve to visualize the performance metric over a
    # # range of values for some hyperparameters (C, gamma, epsilon, etc.)
    # for (param_name, param_range) in tuned_parameters.items():
    #     if 'kernel' not in param_name:
    #         plot_validation_curve(grid.best_estimator_, X, y, param_name, param_range, scorer)
    #
    # # plot learning curve to visualize the effect of the
    # # number of observations on the performance metric
    # plot_learning_curve(grid.best_estimator_, X, y, scorer)
    #
    # # save predictions on the blind test set
    # np.savetxt('optiml/ml/data/ML-CUP19/dmeoli_ML-CUP19-TS.csv', grid.predict(load_ml_cup_blind()), delimiter=',')
