from tensorflow.keras.layers import Dense
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from anytree import Node

from utils import *
from voter import *


class HierarchicalNeuralClassifier:

    def __init__(
            self, units=3, activation='relu',
            optimizer='sgd', optimizer_params={},
            other_rate=.1, regularization=None,
            timeout=10, start=5, max_epochs=20,
            threshold=.2, threshold_ratio=.5,
            validation_split=None, validation_data=None,
            patience=10, batch_size=32,
            loss='categorical_crossentropy', output_activation='sigmoid'):

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
        self.other_rate = other_rate
        self.output_activation = 'sigmoid'

        # специальная метка для класса другое
        self._OTHER_CLASS_ID = -12992929292292

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
            Dense(output_shape[0], activation=self.output_activation)
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
        self.tree = Node(0)
        self.node_to_class = {}
        self.node_counter = 0
        classes = list(np.unique(y))
        self.node_counter += 1
        node = Node(self.node_counter, self.tree)
        self._fit_node(classes, node)
        return self

    def _fit_terminal_node(self, classes, node):
        mask = create_mask(
            self.y, classes, other_rate=self.other_rate)
        y = self.y[mask].copy()
        encoder = OneHotEncoder(categories='auto', sparse=False)
        encoder.fit(y.reshape(-1, 1))
        y = encoder.transform(y.reshape(-1, 1))

        model = self._build_model(
            self.units, self.input_shape, (len(classes),)
        )
        model.fit(self.X[mask], y, epochs=self.max_epochs, verbose=False)
        self.models[node.name] = model

    def _fit_node(self, classes, node):
        print(f"Fitting node with classes {classes}")
        encoder = OneHotEncoder(categories='auto', sparse=False)
        default_classes = classes.copy()

        # КЛАСС ДРУГОЕ НЕ НАХОДИТСЯ В КЛАССАХ РАЗОБРАТЬСЯ!!!!
        mask = create_mask(
            self.y, classes, other_rate=self.other_rate)
        y = self.y[mask].copy()
        encoder.fit(y.reshape(-1, 1))

        old_map = dict(zip(classes, classes))
        model = self._build_model(
            self.units, self.input_shape, (len(classes),)
        )
        stop_flag = False
        epoch = 0

        while not stop_flag and epoch < self.max_epochs:
            if len(y) == 0:
                raise ValueError

            y_one_hot = encoder.transform(y.reshape(-1, 1))
            num_epochs = self.start if epoch == 0 else self.timeout
            epoch += num_epochs

            model.fit(
                self.X[mask], y_one_hot, epochs=num_epochs,
                verbose=False)
            print(f'epoch {epoch}')
            y_pred = model.predict(self.X[mask])
            class_map = perform_voting(
                y, y_pred, classes=classes, default_classes=default_classes,
                threshold=self.threshold,
                threshold_ratio=self.threshold_ratio
            )
            if len(set(class_map.values())) == 2:
                stop_flag = True

            old_map = connect_map(old_map, class_map)
            print('class_map', class_map)
            print('old map', old_map)
            print('unique y before', np.unique(y))
            y = remap(y, class_map)
            classes = list(set(y))
            print('unique y after', classes)

        self.models[node.name] = model

        subsets = get_subsets(old_map)

        for super_class, subset in subsets.items():

            if len(subset) > 2:
                self.node_counter += 1
                self.node_to_class[self.node_counter] = super_class
                new_node  = Node(self.node_counter, parent=node)
                self._fit_node(subset, new_node)
            elif len(subset) == 2:
                self.node_counter += 1
                self.node_to_class[self.node_counter] = super_class
                new_node = Node(self.node_counter, parent=node)
                self._fit_terminal_node(subset, new_node)
            else:
                continue

    def visualize(self):
        pass
