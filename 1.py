import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Загружаем встроенный датасет "Ирисы"
iris = datasets.load_iris()
'''
print("Ключи словаря iris:", iris.keys())
print("Описание датасета:\n", iris['DESCR'][:200], '...')
'''
# X - признаки (числовые характеристики цветов), y - метки классов (виды ирисов)
X = iris.data
y = iris.target
'''
print("Форма матрицы признаков X:", X.shape)
print("Первые 5 строк X:\n", X)
print("Метки классов:", np.unique(y))

for i in range(4):
    for j in range(i):
        plt.scatter(X[:, i], X[:, j], c=y)
        plt.xlabel(f'признак номер {i}')
        plt.ylabel(f'признак номер {j}')
        plt.title('Ирисы: два признака')
        plt.show()
'''
# Разбиваем X и y на обучающую и тестовую части (по 75% и 25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

print("Форма обучающей выборки:", X_train.shape)
print("Форма тестовой выборки:", X_test.shape)

# количество ирисов в тренировочной выборке
n = len(y_train)

# Создаём модель KNN. определим самый оптимальный k (количество соседей) для самого точного идентифицирования ирисов.
max_accuracy = 0
best_k = 0
for k in range(1,n):
    knn = KNeighborsClassifier(n_neighbors=k)

    # Обучаем модель на обучающих данных
    knn.fit(X_train, y_train)

    # Делаем прогноз на тестовых данных
    y_pred = knn.predict(X_test)

    # Оцениваем качество: доля правильных ответов
    accuracy = np.mean(y_pred == y_test)    
    if (accuracy > max_accuracy):
        max_accuracy = accuracy
        best_k = k

# визуализируем наилучшее разделение выборки
knn = KNeighborsClassifier(n_neighbors=best_k)
# Обучаем модель на обучающих данных
knn.fit(X_train, y_train)
# Делаем прогноз на тестовых данных
y_pred = knn.predict(X_test)
# Точность
print(f"Точность на тестовой выборке: {max_accuracy:.2f}")

# визуализируем тренировочные и тестовые ирисы

for i in range(4):
    for j in range(i):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # две оси: слева и справа

        # Слева — тренировочная выборка
        axes[0].scatter(X_train[:, i], X_train[:, j], c=y_train)
        axes[0].set_xlabel(f'признак номер {i}')
        axes[0].set_ylabel(f'признак номер {j}')
        axes[0].set_title('Тренировочная выборка')

        # Справа — тестовая выборка
        axes[1].scatter(X_test[:, i], X_test[:, j], c=y_test)
        axes[1].set_xlabel(f'признак номер {i}')
        axes[1].set_ylabel(f'признак номер {j}')
        axes[1].set_title('Тестовая выборка')

        plt.tight_layout()
        plt.show()