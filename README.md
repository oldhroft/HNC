# HNC
Hierarchical Neural Classifier

Основной класс - HierarchicalNeuralClassifier

```python
from HNC import HierarchicalNeuralClassifier
hnc = HierarchicalNeuralClassifier()
```

### Параметры класса :

units (default=3): число нейронов скрытого слоя

activation (default='relu'): активация скрытого слоя ('relu', 'sigmoid', 'tanh', etc https://keras.io/activations/)

optimizer (default='sgd'): передавать строкой, оптимизатор, см. keras.optimizers https://keras.io/optimizers/

optimizer_params (default={}): параметры оптимизатора, см. keras.optimizers https://keras.io/optimizers/,

regularization (default=None): регуляризация скрытого слоя ('l1', 'l2', или собственная регуляризация, подробнее https://keras.io/regularizers/)

timeout (default=10): число эпох перед слиянием, 

start (default=5): число эпох после слияния,

max_epochs (default=20): максимальное число эпох,

threshold (default=.2): порог слияния по активации, 

threshold_ratio (default=.5): порог слияния по доле проголосовавших примеров,

validation_split (default=None): доля примеров тренировочного набора для использования в качестве валидационного, 

validation_data (default=None): валидационный набор,

patience (default=10): число эпох без уменьшения функции потерь на валидационном набре перед остановкой

batch_size (default=32): размер батча,

loss(default='categorical_crossentropy'): функция потерь, см. https://keras.io/losses/

output_activation(default='sigmoid'): функция активации выходного слоя

### Методы класса

В классе *HierarchicalNeuralClassifier* реализованы методы fit, predict и visualize

Пример использования (классификатор цифр):

```python
import numpy as np
from HNC import HierarchicalNeuralClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# load data
data = load_digits()
X = data['data']
y = data['target']

# build classifier
hnc = HierarchicalNeuralClassifier(
    output_activation='softmax', start=30, threshold=.2, timeout=20,
    max_epochs=230,
)

# scale data
X_scaled = StandardScaler().fit_transform(X)

# fit classifier
hnc.fit(X_scaled, y)

# make inference on train set
preds = hnc.predict(X_scaled)

# get visualization of the resulting tree
tree = hnc.visualize()
print(tree)
```



