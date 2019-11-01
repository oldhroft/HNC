from tensorflow.keras.layers import Dense
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from anytree import Node

from utils import *
from voter import *


class HierarchicalNeuralClassifier:

    def __init__(
            self, units=3, activation='relu',
            optimizer='sgd', optimizer_params={},
            other_rate=.1, regularization=None,
            timeout=10, start=5, max_epochs=20,
            threshold=.2, threshold_ratio=.5, validation_split=None,
            validation_data=None, patience=10, batch_size=32,
            loss='categorical_crossentropy'):

        self.units = units
        self.activation = activation
        self.optimizer_params = optimizer_params
        self.optimizer = optimizer
        self.regularization = regularization
        self.timeout = timeout
        self.start = start
        self.threshold = threshold
        self.threshold_ratio = threshold_ratio
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.patience = patience
        self.batch_size = batch_size
        self.loss = loss
        self.max_epochs = max_epochs

    def _get_optimizer(self):

        if self.optimizer == 'sgd':
            return optimizers.SGD(**self.optimizer_params)
        elif self.optimizer == 'adam':
            return optimizers.Adam(**self.optimizer_params)
        elif self.optimizer == 'adagrad':
            return optimizers.Adagrad(**self.optimizer_params)
        elif self.optimizer == 'adadelta':
            return optimizers.Adadelta(**self.optimizer_params)
        elif self.optimizer == 'adax':
            return optimizers.Adamax(**self.optimizer_params)
        elif self.optimizer == 'nadam':
            return optimizers.Nadam(**self.optimizer_params)
        elif self.optimizer == 'rmsprop':
            return optimizer.RMSprop(**self.optimizer_params)
        else:
            raise NotImplementedError

    def _build_model(self, units, input_shape, output_shape):
        model = Sequential([
            Dense(
                units, input_shape=input_shape,
                activation=self.activation,
                kernel_regularizer=self.regularization,
            ),
            Dense(output_shape[0], activation='sigmoid')
        ])

        model.compile(
            loss=self.loss,
            optimizer=self._get_optimizer()
        )
        return model

    def fit(self, X, y, verbose=1):
        self.models = {}
        self.X = X
        self.y = y
        self.input_shape = (X.shape[1],)
        classes = np.unique(y)
        old_map = dict(zip(classes, classes))

        self.tree = Node(0)
        model = self._build_model(
            self.units, self.input_shape, (len(classes),)
        )
        stop_flag = False
        epoch = 0

        while not stop_flag and epoch < self.max_epochs:
            y_one_hot = to_one_hot(remap(y, old_map))
            num_epochs = self.start if epoch == 0 else self.timeout
            epoch += num_epochs

            model.fit(X, y_one_hot, epochs=num_epochs, verbose=False)
            print(f'epoch {epoch}')
            y_pred = model.predict(X)
            class_map = perform_voting(
                y, y_pred, classes=classes, threshold=self.threshold,
                threshold_ratio=self.threshold_ratio
            )
            if len(set(class_map.values())) == 2:
                stop_flag = True

            old_map = connect_map(old_map, class_map)

        self.models[0] = model
        print(old_map)

        return self

    def fit_node(self, classes, super_class, parent):
        pass
