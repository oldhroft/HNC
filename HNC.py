from tensorflow.keras.layers import Dense
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
from sklearn.preprocessing import OneHotEncoder
from anytree import Node

from numpy import concatenate

from utils import *
from voter import *
from visualization import visualize_tree


class HierarchicalNeuralClassifier:

    def __init__(
            self, units=3, activation='relu',
            optimizer='sgd', optimizer_params={},
            other_rate=.1, regularization=None,
            timeout=10, start=5, max_epochs=20,
            end_fit=10,
            threshold=.2, threshold_ratio=.5,
            validation_split=None,
            patience=10, batch_size=32, verbose=1,
            loss='categorical_crossentropy',
            output_activation='sigmoid', backbone=None):

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
        self.patience = patience
        self.batch_size = batch_size
        self.loss = loss
        self.max_epochs = max_epochs
        self.other_rate = other_rate
        self.output_activation = output_activation
        self.verbose = verbose
        self.backbone = backbone
        self.end_fit = end_fit

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
            raise ValueError(f'No optimizer named {self.optimizer}')

    def _build_model(self, units, output_shape,
                     input_shape=None, backbone=None):

        if input_shape is None and backbone is None:
            raise ValueError(
                'Either backbone or input_shape should be specified')

        if backbone is None:
            model = Sequential([
                Dense(
                    units, input_shape=input_shape,
                    activation=self.activation,
                    kernel_regularizer=self.regularization,
                ),
                Dense(output_shape[0], activation=self.output_activation)
            ])

        else:
            model = Sequential([
                backbone,
                Dense(
                    units, activation=self.activation,
                    kernel_regularizer=self.regularization,
                ),
                Dense(output_shape[0], activation=self.output_activation)
            ])

        model.compile(
            loss=self.loss,
            optimizer=self._get_optimizer()
        )
        return model

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def fit(self, X, y, verbose=1):
        self.models = {}
        self.X = X
        self.y = y
        self.input_shape = (X.shape[1],)
        self.tree = Node(0)
        self.node_to_class = {}
        self.node_to_classes = {}
        self.encoders = {}
        self.node_counter = 0
        classes = list(set(y))
        self.node_to_classes[self.node_counter] = classes
        self._fit_node(classes, self.tree)
        return self

    def _fit_terminal_node(self, classes, node):
        self.print('\n\n', '-' * 50, sep='')
        self.print(f"Fitting terminal node with classes {classes}")
        mask = create_mask(
            self.y, classes, other_rate=self.other_rate)
        y = self.y[mask].copy()
        encoder = OneHotEncoder(categories='auto', sparse=False)
        encoder.fit(y.reshape(-1, 1))
        y = encoder.transform(y.reshape(-1, 1))

        self.encoders[node.name] = encoder

        model = self._build_model(
            self.units, (len(classes),), self.input_shape,
            clone_model(self.backbone)
            if self.backbone is not None else None)
        model.fit(self.X[mask], y, epochs=self.end_fit,
                  verbose=self.verbose > 1, batch_size=self.batch_size)
        self.models[node.name] = model

        for a_class in classes:
            self.node_counter += 1
            self.node_to_class[self.node_counter] = a_class
            Node(self.node_counter, parent=node)
            self.node_to_classes[self.node_counter] = [a_class]

    def _fit_node(self, classes, node):
        self.print('\n\n', '-' * 50, sep='')
        self.print(f"Fitting node with classes {classes}")

        encoder = OneHotEncoder(categories='auto', sparse=False)
        voter = Voter(classes, strategy=self.threshold,
                      threshold_ratio=self.threshold_ratio)

        mask = create_mask(
            self.y, classes, other_rate=self.other_rate)
        y = self.y[mask].copy()
        encoder.fit(y.reshape(-1, 1))
        self.encoders[node.name] = encoder

        old_map = dict(zip(classes, classes))
        model = self._build_model(
            self.units, (len(classes),), self.input_shape,
            clone_model(self.backbone)
            if self.backbone is not None else None)
        voter.build_voter(self.X[mask], model)
        stop_flag = False
        epoch = 0

        while not stop_flag and epoch < self.max_epochs:
            if len(y) == 0:
                raise ValueError('Empty targets')

            y_one_hot = encoder.transform(y.reshape(-1, 1))
            num_epochs = self.start if epoch == 0 else self.timeout
            epoch += num_epochs

            model.fit(
                self.X[mask], y_one_hot, epochs=num_epochs,
                verbose=self.verbose > 1, batch_size=self.batch_size)
            self.print(f'epoch {epoch}')
            y_pred = model.predict(self.X[mask])

            self.print(f'Performing voting, epoch {epoch}')
            class_map = voter.vote(y, y_pred, classes)

            if len(set(class_map.values())) == 2:
                stop_flag = True

            old_map = connect_map(old_map, class_map)
            self.print('New mapping', class_map)
            self.print('Total mapping', old_map)
            old_classes = sorted(list(set(y)))
            y = remap(y, class_map)
            classes = sorted(list(set(y)))
            self.print('{} - > {}'.format(old_classes, classes), '\n')

        y_one_hot = encoder.transform(y.reshape(-1, 1))
        self.print('Performing end fit')
        model.fit(self.X[mask], y_one_hot, epochs=self.end_fit,
                  batch_size=self.batch_size, verbose=self.verbose > 1)
        self.models[node.name] = model

        subsets = get_subsets(old_map)

        for super_class, subset in subsets.items():

            self.node_counter += 1
            self.node_to_class[self.node_counter] = super_class
            self.node_to_classes[self.node_counter] = subset
            new_node = Node(self.node_counter, parent=node)
            if len(subset) > 2:
                self._fit_node(subset, new_node)
            elif len(subset) == 2:
                self._fit_terminal_node(subset, new_node)
            else:
                pass

    def _predict_node(self, x, node):
        if node.is_leaf:
            return np.ones(len(x)) * self.node_to_class[node.name]
        else:

            preds = self.models[node.name].predict(x)
            preds = (preds == preds.max(axis=1, keepdims=1)).astype('int')
            preds = self.encoders[node.name].inverse_transform(preds).ravel()
            ids = np.arange(len(preds))
            stack = []
            ids_stack = []
            for child in node.children:
                mask = np.isin(preds, self.node_to_classes[child.name])
                if mask.sum() > 0:
                    ids_stack.append(ids[mask])
                    stack.append(self._predict_node(x[mask], child))
            ids = concatenate(ids_stack).argsort()
            preds = concatenate(stack)
            return preds[ids]

    def predict(self, X):
        return self._predict_node(X, self.tree)

    def visualize(self, mode='classes'):
        return visualize_tree(self.tree, mode, self.node_to_class,
                              self.node_to_classes)
