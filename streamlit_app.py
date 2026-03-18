import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Анализ Титаника", layout="wide")

# Заголовок
st.title(" Анализ пассажиров Титаника")
st.markdown("---")

# Загрузка данных
@st.cache_data
def load_data():
    # Создаем небольшие тестовые данные, если файл не найден
    try:
        df = pd.read_csv('titanic.csv')
    except:
        # Тестовые данные, если файл не загрузился
        data = {
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22, 38, 26, 35, 35],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Fare': [7.25, 71.28, 7.92, 53.1, 8.05]
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# Боковая панель
st.sidebar.header("Настройки")

# 1. Описательная статистика
st.header(" Описательная статистика")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Информация о данных")
    st.write(f"**Количество строк:** {df.shape[0]}")
    st.write(f"**Количество столбцов:** {df.shape[1]}")
    
    st.subheader("Типы данных")
    dtypes_df = pd.DataFrame({
        'Столбец': df.columns,
        'Тип данных': df.dtypes.values
    })
    st.dataframe(dtypes_df)

with col2:
    st.subheader("Первые строки данных")
    n_rows = st.slider("Сколько строк показать?", min_value=5, max_value=20, value=10)
    st.dataframe(df.head(n_rows))

st.markdown("---")

# 2. Графики
st.header(" Визуализация данных")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Выживаемость", "Возраст", "Классы", "Пол", "Цены"])

# График 1: Выживаемость
with tab1:
    st.subheader("Распределение выживших и погибших")
    fig, ax = plt.subplots()
    survived_counts = df['Survived'].value_counts()
    labels = ['Погиб', 'Выжил']
    colors = ['#ff6b6b', '#4ecdc4']
    ax.bar(labels, survived_counts.values, color=colors)
    st.pyplot(fig)

# График 2: Возраст
with tab2:
    st.subheader("Распределение возраста")
    fig, ax = plt.subplots()
    ax.hist(df['Age'].dropna(), bins=20, color='#95a5a6', edgecolor='black')
    ax.set_xlabel('Возраст')
    ax.set_ylabel('Количество')
    st.pyplot(fig)

# График 3: Классы
with tab3:
    st.subheader("Распределение по классам")
    fig, ax = plt.subplots()
    class_counts = df['Pclass'].value_counts().sort_index()
    ax.bar(['1 класс', '2 класс', '3 класс'], class_counts.values, color=['#3498db', '#2ecc71', '#e74c3c'])
    st.pyplot(fig)

# График 4: Пол
with tab4:
    st.subheader("Распределение по полу")
    fig, ax = plt.subplots()
    sex_counts = df['Sex'].value_counts()
    ax.bar(sex_counts.index, sex_counts.values, color=['#3498db', '#e74c3c'])
    st.pyplot(fig)

# График 5: Цены с выбором класса
with tab5:
    st.subheader("Анализ цен билетов")
    selected_class = st.selectbox("Выберите класс:", [1, 2, 3])
    class_data = df[df['Pclass'] == selected_class]['Fare'].dropna()
    
    fig, ax = plt.subplots()
    ax.hist(class_data, bins=20, color='#9b59b6', edgecolor='black')
    ax.set_xlabel('Цена билета')
    ax.set_ylabel('Количество')
    ax.set_title(f'{selected_class} класс')
    st.pyplot(fig)
    
    st.write(f"Средняя цена: {class_data.mean():.2f}")
