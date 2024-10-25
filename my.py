import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Прогнозы Онлайн: Без регистрации и смс!")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите свой CSV файл и получите прогнозы", type="csv")

# Настройки для обучения
learning_rate = st.number_input("Введите шаг обучения:", min_value=0.01, max_value=1.0, value=0.1, format="%.2f")
epochs = st.number_input("Введите количество эпох:", min_value=1, max_value=10000, value=1000)

class ЛогистическаяРегрессия:
    def __init__(self, learning_rate, n_inputs, epochs=1000):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = np.random.uniform(-1, 1, size=n_inputs)
        self.intercept_ = np.random.uniform(-1, 1)
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        for _ in range(self.epochs):
            y_pred = self.sigmoid(X @ self.coef_ + self.intercept_)
            error = y_pred - y
            grad_coef_ = X.T @ error / len(y)
            grad_intercept = np.sum(error) / len(y)
            self.coef_ -= grad_coef_ * self.learning_rate
            self.intercept_ -= grad_intercept * self.learning_rate
        return self.coef_, self.intercept_

# Обработка загруженного файла
if uploaded_file is not None:
    try:
        # Загрузка и отображение данных
        data = pd.read_csv(uploaded_file)
        st.write("Вот ваш набор данных:")
        st.dataframe(data)

        # Подготовка данных и стандартизация
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Запуск модели
        лог_рег = ЛогистическаяРегрессия(learning_rate=learning_rate, n_inputs=features.shape[1], epochs=epochs)
        coef, intercept = лог_рег.fit(features_scaled, target)
        weights = {col: weight for col, weight in zip(features.columns, coef)}
        weights["Свободный член"] = intercept

        # Вывод коэффициентов регрессии пользовательской модели
        st.write("Коэффициенты регрессии (наша модель):")
        st.json(weights)

        # Сравнение с моделью sklearn
        log_reg_sklearn = LogisticRegression()
        log_reg_sklearn.fit(features_scaled, target)
        
        # Вывод коэффициентов модели sklearn в аналогичном формате
        sklearn_weights = {col: weight for col, weight in zip(features.columns, log_reg_sklearn.coef_[0])}
        sklearn_weights["Свободный член"] = log_reg_sklearn.intercept_[0]
        
        st.write("Коэффициенты модели sklearn:")
        st.json(sklearn_weights)

        # Выбор фичей для графика
        st.subheader("Создаём Scatter Plot")
        x_feature = st.selectbox("Выберите фичу для оси X", options=features.columns)
        y_feature = st.selectbox("Выберите фичу для оси Y", options=features.columns)

        # Построение графика
        if x_feature and y_feature:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=target.name, palette="viridis")
            plt.title(f'Распределение {x_feature} и {y_feature}')
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Что-то пошло не так: {e}")
else:
    st.info("Загрузите CSV файл, чтобы начать магию предсказаний!")
