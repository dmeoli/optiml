from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from yase.ml.neural_network import NeuralNetworkClassifier
from yase.ml.neural_network.activations import softmax, linear, relu
from yase.ml.neural_network.layers import Conv2D, MaxPool2D, Flatten, FullyConnected
from yase.ml.neural_network.losses import sparse_categorical_cross_entropy, categorical_cross_entropy
from yase.ml.utils import plot_model_loss, plot_model_accuracy
from yase.optimization.unconstrained.stochastic import Adam

if __name__ == '__main__':
    # ub = np.array([11.2588947455727,
    #                11.6231677483025,
    #                8.50794726517403,
    #                11.6535034245561,
    #                10.5294369849016,
    #                8.39016161999764,
    #                9.11399287546819,
    #                10.1875260768199,
    #                11.8300273417372,
    #                11.8595541407971])
    #
    # q = np.array([-1.02502047805129,
    #               -0.882082762438178,
    #               -0.717461923979454,
    #               -0.903270054056326,
    #               -0.851787333699129,
    #               -0.960644560327484,
    #               -0.988679397784216,
    #               -0.957816183856348,
    #               -0.778208380884683,
    #               -0.785843035645380])
    #
    # Q = np.array([[0.0545809922930021, 0.0402299676792409, 0.0260859172560283, 0.0318396466688350, 0.0395665097221708,
    #                0.0311522042513846, 0.0344816391971015, 0.0306682107083224, 0.0285997960344617, 0.0322126170453263],
    #               [0.0402299676792409, 0.0409409354751398, 0.0205084927646861, 0.0278804939896159, 0.0298183840424571,
    #                0.0266563245159146, 0.0274077182577349, 0.0251053587475510, 0.0239111113867816, 0.0217976416677582],
    #               [0.0260859172560283, 0.0205084927646861, 0.0306530124733982, 0.0250396974749879, 0.0258126773789457,
    #                0.0172530896280471, 0.0247905358399176, 0.0180253681038746, 0.0197543169196889, 0.0250395752646856],
    #               [0.0318396466688350, 0.0278804939896159, 0.0250396974749879, 0.0344509618632178, 0.0270121458517153,
    #                0.0277453283629864, 0.0282021443435733, 0.0255948023580215, 0.0192856760053380, 0.0285060123463953],
    #               [0.0395665097221708, 0.0298183840424571, 0.0258126773789457, 0.0270121458517153, 0.0369403437918012,
    #                0.0220567053969042, 0.0295967053217025, 0.0249383427556326, 0.0219204604142792, 0.0253599346776480],
    #               [0.0311522042513846, 0.0266563245159146, 0.0172530896280471, 0.0277453283629864, 0.0220567053969042,
    #                0.0364549307902831, 0.0241631894501881, 0.0291686669216518, 0.0212624001233939, 0.0229679129099219],
    #               [0.0344816391971015, 0.0274077182577349, 0.0247905358399176, 0.0282021443435733, 0.0295967053217025,
    #                0.0241631894501881, 0.0350984165785865, 0.0248496543006175, 0.0229442401664358, 0.0268999199317812],
    #               [0.0306682107083224, 0.0251053587475510, 0.0180253681038746, 0.0255948023580215, 0.0249383427556326,
    #                0.0291686669216518, 0.0248496543006175, 0.0323861881459086, 0.0185077147476050, 0.0226591682006490],
    #               [0.0285997960344617, 0.0239111113867816, 0.0197543169196889, 0.0192856760053380, 0.0219204604142792,
    #                0.0212624001233939, 0.0229442401664358, 0.0185077147476050, 0.0244877805048506, 0.0165918338970937],
    #               [0.0322126170453263, 0.0217976416677582, 0.0250395752646856, 0.0285060123463953, 0.0253599346776480,
    #                0.0229679129099219, 0.0268999199317812, 0.0226591682006490, 0.0165918338970937, 0.0368086483534745]])

    # print(ProjectedGradient(BoxConstrainedQuadratic(Q, q, ub), verbose=False).minimize()[1])
    # print(InteriorPoint(BoxConstrainedQuadratic(Q, q, ub), verbose=False).minimize()[1])
    # print(ActiveSet(BoxConstrainedQuadratic(Q, q, ub), verbose=False).minimize()[1])
    # print(FrankWolfe(BoxConstrainedQuadratic(Q, q, ub), verbose=False).minimize()[1])
    # print(LagrangianDual(BoxConstrainedQuadratic(Q, q, ub), verbose=False).minimize()[1])

    # TODO
    #  - capire dov'è possibile aggiungere dei random states seed nelle utils (generatori di dati)
    #  - capire perchè la soluzione della linear regression regolarizzata è diversa dalla closed
    #    solution anche con il metodo di newton o bfgs ed aggiungere un test
    #    per il ridge perceptron (aka ridge regression)
    #  - fixare schedules per gradient descent
    #  - fixare la gestione dei batch negli algoritmi stocastici plottando la media della loss per
    #    ogni epoch ed implementare l'early stopping come in sklearn
    #  - fixare gli initializers per le reti neurali passandogli il random state seed dalla neural network
    #    (verificare che funzionino con un test sia per i generatodi di dati che per gli initializers dei vettori di pesi)
    #  - fare dei test per i multi-label per capire come funziona per MLPClassifier ed anche su SVM
    #    (eliminare i TO DO in giro) e poi capire bene il ruolo di binary cross entropy e di categorical cross entropy
    #  - considerare l'idea di fare i plot delle loss più carine con i puntini alpha=0.2 per evidenziare i batch sizes
    #  - capire perchè l'algoritmo SMO custom ha qualche problema nella grid search rispetto a sklean (?)
    #    ed aggiungere delle checkbox TO DO con "aprire la issue per la versione pesata di C come in sklearn"
    #  - aggiungere proximal bundle al ipynb capendo dove inserirlo se nei line search o nei stochastic
    #  - aggiungere file ipynb per Constrained Optimizers con plot delle box dei vincoli
    #  - capire come mantenere la forward tra function e jacobian per non ricalcolarla, magari una cache,
    #    e c'è la stessa inefficienza nella funzione lagrangiana
    #  - prendere la doc da climin più possibile ed aggiungerla ai jupyter con pseudocodice anche dal pdf delle slides
    #  - rivedere il codice per generare i bcqp (corregere quello esistente come in matlab)
    #  - modellare dei metodi comuni per gli accelleratori in Stochastic Optimizer (Nesterov e Standard)
    #    o come in climin con un metodo _iterate e poi tutto il resto generalizzato più possibile
    #  - riaggiungere l'accellerated gradient di Frangioni con la line search ? in caso aggiungerla anche nel ipynb

    X, y = fetch_openml('mnist_784', return_X_y=True)
    # normalizing the RGB codes by dividing it to the max RGB value
    X = X / 255

    # splitting into train and test data
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    # reshaping the images data to a tensor of shape
    # (n_samples, image_height, image_width, n_channels)
    # for convolution layers
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

    y_train = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(-1, 1))

    cnn = NeuralNetworkClassifier(
        (Conv2D(in_channels=1, out_channels=6, kernel_size=(5, 5), strides=(1, 1),
                padding='same', channels_last=True, activation=relu),  # => (n_samples, 28, 28, 6)
         MaxPool2D(pool_size=(2, 2), strides=(2, 2)),  # => (n_samples, 14, 14, 6)
         Conv2D(in_channels=6, out_channels=16, kernel_size=(5, 5), strides=(1, 1),
                padding='same', channels_last=True, activation=relu),  # => (n_samples, 14, 14, 16)
         MaxPool2D(pool_size=(2, 2), strides=(2, 2)),  # => (n_samples, 7, 7, 16)
         Flatten(),  # => (n_samples, 7 * 7 * 16)
         FullyConnected(n_in=7 * 7 * 16, n_out=10, activation=softmax)),
        loss=categorical_cross_entropy, optimizer=Adam, learning_rate=0.002,
        momentum_type='nesterov', momentum=0.6, max_iter=10, batch_size=64, verbose=True)
    cnn.fit(X_train, y_train)
    plot_model_loss(cnn.loss_history)
    plot_model_accuracy(cnn.accuracy_history)
    print(classification_report(y_test, cnn.predict(X_test)))

    # from sklearn.metrics import make_scorer
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.multioutput import MultiOutputRegressor
    #
    # from utils import load_ml_cup, mean_euclidean_error, load_ml_cup_blind, plot_validation_curve, plot_learning_curve
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
    # np.savetxt('yase/ml/data/ML-CUP19/dmeoli_ML-CUP19-TS.csv', grid.predict(load_ml_cup_blind()), delimiter=',')
