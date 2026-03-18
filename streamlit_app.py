import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Настройка страницы
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# Заголовок
st.title("🚢 Titanic Passenger Analysis")
st.markdown("---")

# Загрузка данных
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('titanic.csv')
    except:
        # Тестовые данные, если файл не загрузился
        data = {
            'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
            'Name': ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen', 'Moran', 'McCarthy', 'Palsson', 'Johnson', 'Nasser'],
            'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
            'Age': [22, 38, 26, 35, 35, None, 54, 2, 27, 14],
            'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07]
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# Боковая панель
st.sidebar.header("Settings")
st.sidebar.info("Use the tabs below to explore different aspects of the data")

# 1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
st.header("📊 Descriptive Statistics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Information")
    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")
    
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values
    })
    st.dataframe(dtypes_df, use_container_width=True)

with col2:
    st.subheader("Preview Data")
    n_rows = st.selectbox("Number of rows to display:", [5, 10, 15, 20, 30, 50], index=1)
    st.dataframe(df.head(n_rows), use_container_width=True)

st.markdown("---")

# 2. ГРАФИКИ
st.header("📈 Data Visualization")

# Создаем вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Survival", "Age", "Class", "Gender", "Fare"])

# ГРАФИК 1: Выживаемость
with tab1:
    st.subheader("Survival Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        survived_counts = df['Survived'].value_counts().sort_index()
        labels = ['Died (0)', 'Survived (1)']
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax.bar(labels, survived_counts.values, color=colors)
        ax.set_ylabel('Number of Passengers')
        ax.set_title('Survival Count')
        
        # Добавляем цифры на столбцы
        for bar, count in zip(bars, survived_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(df)*100:.1f}%)',
                    ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Statistics")
        survival_rate = df['Survived'].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
        st.metric("Total Passengers", len(df))
        st.metric("Survived", len(df[df['Survived']==1]))
        st.metric("Died", len(df[df['Survived']==0]))

# ГРАФИК 2: Возраст
with tab2:
    st.subheader("Age Distribution")
    
    # Убираем пустые значения
    age_data = df['Age'].dropna()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(age_data, bins=20, color='#95a5a6', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Age')
        ax.set_ylabel('Number of Passengers')
        ax.set_title('Age Distribution')
        ax.axvline(age_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age_data.mean():.1f}')
        ax.legend()
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Age Statistics")
        st.metric("Mean Age", f"{age_data.mean():.1f}")
        st.metric("Median Age", f"{age_data.median():.1f}")
        st.metric("Min Age", f"{age_data.min():.1f}")
        st.metric("Max Age", f"{age_data.max():.1f}")
        st.metric("Missing Ages", df['Age'].isna().sum())

# ГРАФИК 3: Классы
with tab3:
    st.subheader("Passenger Class Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        class_counts = df['Pclass'].value_counts().sort_index()
        class_labels = ['1st Class', '2nd Class', '3rd Class']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax.bar(class_labels, class_counts.values, color=colors)
        ax.set_ylabel('Number of Passengers')
        ax.set_title('Passenger Class Distribution')
        
        # Добавляем цифры
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    str(count), ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Class Statistics")
        for i, label in enumerate(class_labels):
            count = class_counts.iloc[i]
            survival = df[df['Pclass'] == i+1]['Survived'].mean() * 100
            st.metric(f"{label}", f"{count} passengers", f"{survival:.1f}% survived")

# ГРАФИК 4: Пол
with tab4:
    st.subheader("Gender Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sex_counts = df['Sex'].value_counts()
        colors = ['#3498db', '#e74c3c'] if 'male' in sex_counts.index else ['#e74c3c', '#3498db']
        
        bars = ax.bar(sex_counts.index, sex_counts.values, color=colors)
        ax.set_ylabel('Number of Passengers')
        ax.set_title('Gender Distribution')
        
        for bar, count in zip(bars, sex_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    str(count), ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Процент выживания по полу
        survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
        
        bars = ax.bar(survival_by_sex.index, survival_by_sex.values, color=colors)
        ax.set_ylabel('Survival Rate (%)')
        ax.set_title('Survival Rate by Gender')
        ax.set_ylim(0, 100)
        
        for bar, rate in zip(bars, survival_by_sex.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)

# ГРАФИК 5: Цены (реагирует на пользователя)
with tab5:
    st.subheader("Fare Analysis by Class")
    
    # Выбор класса
    selected_class = st.radio("Select Passenger Class:", [1, 2, 3], horizontal=True)
    
    # Фильтруем данные
    class_data = df[df['Pclass'] == selected_class]['Fare'].dropna()
    
    if len(class_data) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(class_data, bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Fare')
            ax.set_ylabel('Number of Passengers')
            ax.set_title(f'Fare Distribution - Class {selected_class}')
            ax.axvline(class_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {class_data.mean():.2f}')
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Fare Statistics")
            st.metric("Mean Fare", f"${class_data.mean():.2f}")
            st.metric("Median Fare", f"${class_data.median():.2f}")
            st.metric("Min Fare", f"${class_data.min():.2f}")
            st.metric("Max Fare", f"${class_data.max():.2f}")
            st.metric("Passengers", len(class_data))
    else:
        st.warning(f"No fare data available for Class {selected_class}")

# Дополнительная информация
st.markdown("---")
st.header("📋 Summary Statistics")

if st.checkbox("Show detailed statistics"):
    st.subheader("Survival Statistics by Category")
    
    # Создаем сводную таблицу
    summary_data = []
    
    # Overall
    summary_data.append({
        'Category': 'Overall',
        'Total': len(df),
        'Survived': len(df[df['Survived']==1]),
        'Survival Rate': f"{df['Survived'].mean()*100:.1f}%"
    })
    
    # By Sex
    for sex in df['Sex'].unique():
        sex_df = df[df['Sex'] == sex]
        summary_data.append({
            'Category': f'Sex: {sex}',
            'Total': len(sex_df),
            'Survived': len(sex_df[sex_df['Survived']==1]),
            'Survival Rate': f"{sex_df['Survived'].mean()*100:.1f}%"
        })
    
    # By Class
    for pclass in sorted(df['Pclass'].unique()):
        class_df = df[df['Pclass'] == pclass]
        summary_data.append({
            'Category': f'Class: {pclass}',
            'Total': len(class_df),
            'Survived': len(class_df[class_df['Survived']==1]),
            'Survival Rate': f"{class_df['Survived'].mean()*100:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

# Подвал
st.markdown("---")
st.markdown(" **Titanic Passenger Analysis Dashboard** | Created with Streamlit")
