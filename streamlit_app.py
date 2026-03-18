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
    
    # Проверяем, что есть оба значения (0 и 1)
    survived_dict = {0: 0, 1: 0}
    for idx, val in survived_counts.items():
        survived_dict[idx] = val
    
    labels = ['Погиб', 'Выжил']
    colors = ['#ff6b6b', '#4ecdc4']
    ax.bar(labels, [survived_dict[0], survived_dict[1]], color=colors)
    ax.set_ylabel('Количество')
    st.pyplot(fig)

# График 2: Возраст
with tab2:
    st.subheader("Распределение возраста")
    fig, ax = plt.subplots()
    age_data = df['Age'].dropna()
    if len(age_data) > 0:
        ax.hist(age_data, bins=20, color='#95a5a6', edgecolor='black')
        ax.set_xlabel('Возраст')
        ax.set_ylabel('Количество')
    else:
        ax.text(0.5, 0.5, 'Нет данных о возрасте', ha='center')
    st.pyplot(fig)

# График 3: Классы (ИСПРАВЛЕНО!)
with tab3:
    st.subheader("Распределение по классам")
    fig, ax = plt.subplots()
    
    # Получаем количество для каждого класса и гарантируем наличие всех трёх
    class_counts = df['Pclass'].value_counts()
    
    # Создаем словарь со всеми классами (1, 2, 3)
    all_classes = {1: 0, 2: 0, 3: 0}
    for idx, val in class_counts.items():
        if idx in [1, 2, 3]:  # Только допустимые классы
            all_classes[idx] = val
    
    # Преобразуем в список для графика
    counts_list = [all_classes[1], all_classes[2], all_classes[3]]
    
    ax.bar(['1 класс', '2 класс', '3 класс'], counts_list, 
           color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel('Количество')
    
    # Добавляем подписи значений
    for i, v in enumerate(counts_list):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    st.pyplot(fig)

# График 4: Пол
with tab4:
    st.subheader("Распределение по полу")
    fig, ax = plt.subplots()
    sex_counts = df['Sex'].value_counts()
    
    # Проверяем, что есть оба значения
    sex_dict = {'male': 0, 'female': 0}
    for idx, val in sex_counts.items():
        sex_dict[idx] = val
    
    labels = ['Мужчины', 'Женщины']
    ax.bar(labels, [sex_dict['male'], sex_dict['female']], 
           color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Количество')
    st.pyplot(fig)

# График 5: Цены с выбором класса (реагирует на пользователя)
with tab5:
    st.subheader("Анализ цен билетов")
    
    # Получаем уникальные классы, которые есть в данных
    available_classes = sorted(df['Pclass'].unique())
    if len(available_classes) == 0:
        available_classes = [1, 2, 3]
    
    selected_class = st.selectbox("Выберите класс:", available_classes)
    
    # Фильтруем данные по выбранному классу
    class_data = df[df['Pclass'] == selected_class]['Fare'].dropna()
    
    fig, ax = plt.subplots()
    
    if len(class_data) > 0:
        ax.hist(class_data, bins=20, color='#9b59b6', edgecolor='black')
        ax.set_xlabel('Цена билета')
        ax.set_ylabel('Количество')
        ax.set_title(f'{selected_class} класс')
        
        # Добавляем статистику
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Средняя цена", f"{class_data.mean():.2f}")
        with col2:
            st.metric("Минимальная цена", f"{class_data.min():.2f}")
        with col3:
            st.metric("Максимальная цена", f"{class_data.max():.2f}")
    else:
        ax.text(0.5, 0.5, f'Нет данных о ценах для {selected_class} класса', ha='center')
    
    st.pyplot(fig)

st.markdown("---")
st.markdown(" Дашборд для анализа данных пассажиров Титаника")
