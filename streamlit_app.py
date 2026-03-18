import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Анализ пассажиров Титаника", layout="wide")

# Заголовок
st.title(" Анализ данных пассажиров Титаника")
st.markdown("---")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    return df

df = load_data()

# БОКОВАЯ ПАНЕЛЬ (для пользовательского ввода)
st.sidebar.header("Настройки")

# 1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
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

# 2. ГРАФИКИ
st.header(" Визуализация данных")

# Создаем вкладки для разных графиков
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Выживаемость", "Возраст", "Классы", "Пол", "Цены"])

# ВКЛАДКА 1: Столбчатая диаграмма - Выживаемость
with tab1:
    st.subheader("Распределение выживших и погибших")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    survived_counts = df['Survived'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    labels = ['Погиб', 'Выжил']
    
    bars = ax.bar(labels, survived_counts.values, color=colors)
    ax.set_ylabel('Количество')
    
    # Добавляем цифры на столбцы
    for bar, count in zip(bars, survived_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom')
    
    st.pyplot(fig)

# ВКЛАДКА 2: Гистограмма - Возраст
with tab2:
    st.subheader("Распределение возраста пассажиров")
    
    # Убираем пустые значения возраста
    age_data = df['Age'].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(age_data, bins=30, color='#95a5a6', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Возраст')
    ax.set_ylabel('Количество')
    ax.axvline(age_data.mean(), color='red', linestyle='--', label=f'Средний возраст: {age_data.mean():.1f}')
    ax.legend()
    
    st.pyplot(fig)

# ВКЛАДКА 3: Круговая диаграмма - Классы
with tab3:
    st.subheader("Распределение по классам")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    class_counts = df['Pclass'].value_counts().sort_index()
    labels = ['1 класс', '2 класс', '3 класс']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(class_counts.values, 
                                        labels=labels, 
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        startangle=90)
    
    st.pyplot(fig)

# ВКЛАДКА 4: Столбчатая диаграмма - Пол
with tab4:
    st.subheader("Распределение по полу")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Общее распределение по полу
    sex_counts = df['Sex'].value_counts()
    axes[0].bar(sex_counts.index, sex_counts.values, color=['#3498db', '#e74c3c'])
    axes[0].set_title('Все пассажиры')
    axes[0].set_ylabel('Количество')
    
    # Выживаемость по полу
    survived_sex = df.groupby('Sex')['Survived'].mean() * 100
    axes[1].bar(survived_sex.index, survived_sex.values, color=['#3498db', '#e74c3c'])
    axes[1].set_title('Процент выживших по полу')
    axes[1].set_ylabel('Процент выживших (%)')
    
    plt.tight_layout()
    st.pyplot(fig)

# ВКЛАДКА 5: График, реагирующий на пользовательский ввод
with tab5:
    st.subheader("Анализ цен билетов")
    
    # Выбор класса для анализа
    selected_class = st.selectbox("Выберите класс для анализа:", [1, 2, 3])
    
    # Фильтруем данные
