import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import re

# Импортируем наши модули
from src.data_loader import DataLoader
from src.preprocessor import StudentDataPreprocessor
from src.model import DropoutPredictor
from src.evaluator import ModelEvaluator
from src.monitoring import PerformanceMonitor, StudentRiskTracker
from src.reporting import ReportGenerator
import src.utils as utils

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создаем директории
utils.create_directory_structure()

# Функция для загрузки данных
def load_data(file_path=None, generate_sample=False, sample_size=1000):
    loader = DataLoader()
    
    if generate_sample:
        data = loader.generate_sample_data(sample_size)
        st.success(f"Сгенерированы тестовые данные с {sample_size} записями")
    elif file_path:
        data = loader.load_local_file(file_path)
        st.success(f"Загружены данные из {file_path}")
    else:
        data = None
        
    return data

# Функция для предобработки данных
def preprocess_data(data, target_column='dropout'):
    preprocessor = StudentDataPreprocessor()
    
    # Создаем дополнительные признаки
    data_enriched = preprocessor.create_additional_features(data)
    
    # Преобразуем данные
    X, y = preprocessor.fit_transform(data_enriched, target_column)
    
    return preprocessor, X, y, data_enriched

# Функция для обучения модели
def train_model(X, y, model_name, hyperparameter_tuning=False):
    predictor = DropoutPredictor()
    
    with st.spinner(f"Обучение модели {model_name}..."):
        start_time = time.time()
        metrics = predictor.train(X, y, model_name=model_name, 
                                 hyperparameter_tuning=hyperparameter_tuning)
        training_time = time.time() - start_time
    
    st.success(f"Модель {model_name} обучена за {training_time:.2f} сек.")
    
    return predictor, metrics

# Функция для отображения метрик
def display_metrics(metrics):
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")
    
    # Дополнительные метрики
    if 'cv_accuracy_mean' in metrics:
        st.info(f"Результаты кросс-валидации (CV=5): "
               f"Accuracy = {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    
    # Матрица ошибок
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Предсказанные значения')
        ax.set_ylabel('Истинные значения')
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        
    # ROC кривая если есть
    if 'roc_curve' in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        fpr = np.array(metrics['roc_curve']['fpr'])
        tpr = np.array(metrics['roc_curve']['tpr'])
        auc = metrics['roc_curve']['auc']
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

# Функция для визуализации важности признаков
def display_feature_importance(predictor, feature_names):
    feature_importance = predictor.get_feature_importances(feature_names)
    
    if feature_importance is not None:
        st.subheader("Важность признаков")
        
        # Отображаем таблицу важности признаков
        st.dataframe(feature_importance.head(20))
        
        # Строим график
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_col = 'importance' if 'importance' in feature_importance.columns else 'coefficient'
        
        # Берем топ-15 признаков
        top_features = feature_importance.head(15)
        sns.barplot(x=top_features[importance_col], y=top_features['feature'], palette='viridis', ax=ax)
        ax.set_title('Топ-15 важных признаков')
        ax.set_xlabel('Важность')
        ax.set_ylabel('Признак')
        st.pyplot(fig)

# Функция для анализа рисков отчисления
def analyze_student_risks(predictor, data, X, threshold=0.5):
    st.subheader("Анализ рисков отчисления студентов")
    
    # Получаем предсказания
    predictions, probabilities = predictor.predict(X)
    
    # Создаем DataFrame с результатами
    if probabilities is not None:
        if 'student_id' in data.columns:
            student_ids = data['student_id']
        else:
            student_ids = [f"Student_{i}" for i in range(len(predictions))]
            
        results = pd.DataFrame({
            'student_id': student_ids,
            'dropout_probability': probabilities,
            'prediction': predictions,
            'risk_level': pd.cut(
                probabilities, 
                bins=[0, 0.3, 0.6, 1.0], 
                labels=['Низкий', 'Средний', 'Высокий']
            )
        })
        
        # Отображаем статистику
        st.write(f"Общее количество студентов: {len(results)}")
        risk_counts = results['risk_level'].value_counts()
        st.write(f"Студентов с высоким риском отчисления: {risk_counts.get('Высокий', 0)} "
                f"({risk_counts.get('Высокий', 0) / len(results) * 100:.1f}%)")
        
        # Визуализация распределения вероятностей
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results['dropout_probability'], bins=20, kde=True, ax=ax)
        ax.axvline(x=threshold, color='red', linestyle='--', 
                  label=f'Порог ({threshold})')
        ax.set_xlabel('Вероятность отчисления')
        ax.set_ylabel('Количество студентов')
        ax.set_title('Распределение вероятностей отчисления')
        ax.legend()
        st.pyplot(fig)
        
        # Отображаем данные о студентах с высоким риском
        st.subheader("Студенты с высоким риском отчисления")
        high_risk = results[results['risk_level'] == 'Высокий'].sort_values(
            by='dropout_probability', ascending=False)
        
        if not high_risk.empty:
            st.dataframe(high_risk)
            
            # Возможность скачать результаты
            csv = high_risk.to_csv(index=False)
            st.download_button(
                label="Скачать данные о студентах с высоким риском",
                data=csv,
                file_name='high_risk_students.csv',
                mime='text/csv',
            )
        else:
            st.info("Студентов с высоким риском отчисления не найдено")
        
        # Сохраняем оценку рисков в систему отслеживания
        save_assessment = st.checkbox("Сохранить оценку рисков для дальнейшего анализа", value=True)
        
        if save_assessment:
            tracker = StudentRiskTracker()
            notes = st.text_area("Примечания к оценке (опционально)")
            
            if st.button("Сохранить оценку рисков"):
                model_version = predictor.model_name if hasattr(predictor, 'model_name') else "unknown_model"
                tracker.add_risk_assessment(
                    assessment_date=datetime.now(),
                    student_risks=results,
                    model_version=model_version,
                    notes=notes if notes else None
                )
                st.success("Оценка рисков сохранена в системе отслеживания")
        
        return results
    else:
        st.warning("Для этой модели недоступны вероятности прогнозов")
        return None

# Настройка интерфейса приложения
st.set_page_config(
    page_title="Система прогнозирования отчисления студентов",
    page_icon="🎓",
    layout="wide"
)

# Заголовок приложения
st.title("Система прогнозирования отчисления студентов")
st.markdown("Интеллектуальный сервис для прогнозирования отчисления студентов Московского университета имени С.Ю. Витте")

# Боковая панель
st.sidebar.title("Управление")

# Выбор режима работы
mode = st.sidebar.selectbox(
    "Выберите режим работы", 
    ["Загрузка данных", "Обучение модели", "Прогнозирование отчисления", "Мониторинг и анализ", "О системе"]
)

# Сессионные переменные для хранения состояния
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None

# Обработка различных режимов
if mode == "Загрузка данных":
    st.header("Загрузка данных")
    
    # Варианты загрузки данных
    data_source = st.radio(
        "Выберите источник данных",
        ["Загрузить файл", "Загрузить с Kaggle", "Загрузить из облачного хранилища", "Сгенерировать тестовые данные"]
    )
    
    if data_source == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл с данными", 
                                        type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            # Сохраняем файл
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Загружаем данные
            file_path = os.path.join("data", uploaded_file.name)
            st.session_state.data = load_data(file_path=file_path)
    
    elif data_source == "Загрузить с Kaggle":
        st.subheader("Загрузка данных с Kaggle")
        
        # Поле для ввода имени датасета
        kaggle_dataset = st.text_input(
            "Введите имя датасета в формате 'username/dataset-name'",
            placeholder="например, 'ronitf/heart-disease-uci'"
        )
        
        # Информация о настройке API ключа Kaggle
        st.info("""
        Для загрузки данных с Kaggle необходимо настроить API ключ:
        1. Создайте аккаунт на [Kaggle](https://www.kaggle.com)
        2. Перейдите в раздел "Account" -> "Create New API Token"
        3. Скачайте файл kaggle.json и поместите его в папку ~/.kaggle/ (Linux/Mac) или C:\\Users\\YOUR_USERNAME\\.kaggle\\ (Windows)
        """)
        
        # Кнопка для загрузки
        if st.button("Загрузить датасет с Kaggle") and kaggle_dataset:
            try:
                with st.spinner(f"Загрузка датасета {kaggle_dataset}..."):
                    loader = DataLoader()
                    path = loader.download_from_kaggle(kaggle_dataset)
                    
                    # Находим файлы датасета
                    kaggle_files = [f for f in os.listdir(path) if f.endswith(('.csv', '.xlsx', '.xls'))]
                    
                    if kaggle_files:
                        selected_file = st.selectbox("Выберите файл для загрузки", kaggle_files)
                        file_path = os.path.join(path, selected_file)
                        st.session_state.data = load_data(file_path=file_path)
                    else:
                        st.error("Не найдены поддерживаемые форматы файлов в загруженном датасете")
            except Exception as e:
                st.error(f"Ошибка при загрузке датасета: {str(e)}")
    
    elif data_source == "Загрузить из облачного хранилища":
        st.subheader("Загрузка данных из облачного хранилища")
        
        cloud_option = st.selectbox(
            "Выберите облачное хранилище",
            ["Google Drive", "Dropbox", "Microsoft OneDrive", "Amazon S3"]
        )
        
        if cloud_option == "Google Drive":
            drive_link = st.text_input(
                "Введите ссылку на файл в Google Drive",
                placeholder="https://drive.google.com/file/d/..."
            )
            
            st.info("""
            Для доступа к файлу убедитесь, что он открыт для доступа по ссылке.
            URL должен быть в формате https://drive.google.com/file/d/{file_id}/view
            """)
            
            if st.button("Загрузить с Google Drive") and drive_link:
                try:
                    with st.spinner("Загрузка файла с Google Drive..."):
                        # Извлекаем file_id из URL
                        file_id_match = re.search(r'\/d\/([^\/]+)', drive_link)
                        
                        if file_id_match:
                            file_id = file_id_match.group(1)
                            
                            # Загружаем файл
                            loader = DataLoader()
                            file_path = loader.download_from_google_drive(file_id)
                            
                            # Загружаем данные
                            st.session_state.data = load_data(file_path=file_path)
                            st.success(f"Файл успешно загружен из Google Drive")
                        else:
                            st.error("Неверный формат ссылки Google Drive. Убедитесь, что URL содержит '/d/{file_id}'")
                except Exception as e:
                    st.error(f"Ошибка при загрузке данных с Google Drive: {str(e)}")
        
        elif cloud_option == "Dropbox":
            dropbox_link = st.text_input(
                "Введите ссылку на файл в Dropbox",
                placeholder="https://www.dropbox.com/..."
            )
            
            st.info("""
            Убедитесь, что файл открыт для общего доступа.
            URL должен начинаться с https://www.dropbox.com/
            """)
            
            if st.button("Загрузить с Dropbox") and dropbox_link:
                try:
                    with st.spinner("Загрузка файла с Dropbox..."):
                        # Загружаем файл
                        loader = DataLoader()
                        file_path = loader.download_from_dropbox(dropbox_link)
                        
                        # Загружаем данные
                        st.session_state.data = load_data(file_path=file_path)
                        st.success(f"Файл успешно загружен из Dropbox")
                except Exception as e:
                    st.error(f"Ошибка при загрузке данных с Dropbox: {str(e)}")
        
        elif cloud_option == "Microsoft OneDrive":
            onedrive_link = st.text_input(
                "Введите ссылку на файл в OneDrive",
                placeholder="https://1drv.ms/..."
            )
            
            st.info("""
            Убедитесь, что файл открыт для общего доступа.
            URL может начинаться с https://1drv.ms/ или https://onedrive.live.com/
            """)
            
            if st.button("Загрузить с OneDrive") and onedrive_link:
                try:
                    with st.spinner("Загрузка файла с OneDrive..."):
                        # Загружаем файл
                        loader = DataLoader()
                        file_path = loader.download_from_onedrive(onedrive_link)
                        
                        # Загружаем данные
                        st.session_state.data = load_data(file_path=file_path)
                        st.success(f"Файл успешно загружен из OneDrive")
                except Exception as e:
                    st.error(f"Ошибка при загрузке данных с OneDrive: {str(e)}")
        
        elif cloud_option == "Amazon S3":
            st.subheader("Загрузка данных с Amazon S3")
            
            # Создаем две колонки для более компактного интерфейса
            col1, col2 = st.columns(2)
            
            with col1:
                bucket_name = st.text_input("Bucket Name", placeholder="my-bucket")
                object_key = st.text_input("Object Key", placeholder="path/to/my/file.csv")
                aws_region = st.selectbox("AWS Region", ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-northeast-1"])
            
            with col2:
                aws_access_key = st.text_input("AWS Access Key ID")
                aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
            
            if st.button("Загрузить с Amazon S3") and bucket_name and object_key and aws_access_key and aws_secret_key:
                try:
                    with st.spinner("Загрузка файла с Amazon S3..."):
                        # Загружаем файл
                        loader = DataLoader()
                        file_path = loader.download_from_aws_s3(
                            bucket_name=bucket_name,
                            object_key=object_key,
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key,
                            region_name=aws_region
                        )
                        
                        # Загружаем данные
                        st.session_state.data = load_data(file_path=file_path)
                        st.success(f"Файл успешно загружен из Amazon S3")
                except Exception as e:
                    st.error(f"Ошибка при загрузке данных с Amazon S3: {str(e)}")
    
    else:  # Генерация тестовых данных
        sample_size = st.slider("Количество записей", 100, 5000, 1000)
        
        if st.button("Сгенерировать данные"):
            st.session_state.data = load_data(generate_sample=True, sample_size=sample_size)
    
    # Отображаем загруженные данные
    if st.session_state.data is not None:
        st.header("Просмотр данных")
        st.dataframe(st.session_state.data.head(100))
        
        st.markdown(f"**Размер данных:** {st.session_state.data.shape[0]} строк, {st.session_state.data.shape[1]} столбцов")
        
        # Информация о целевой переменной
        if 'dropout' in st.session_state.data.columns:
            dropout_counts = st.session_state.data['dropout'].value_counts()
            dropout_rate = dropout_counts.get(1, 0) / len(st.session_state.data) * 100
            
            st.info(f"Доля отчисленных студентов: {dropout_rate:.2f}% ({dropout_counts.get(1, 0)} из {len(st.session_state.data)})")
            
            # Визуализация распределения целевой переменной
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='dropout', data=st.session_state.data, ax=ax)
            ax.set_xlabel('Отчислен')
            ax.set_ylabel('Количество студентов')
            ax.set_title('Распределение целевой переменной')
            ax.set_xticklabels(['Не отчислен', 'Отчислен'])
            st.pyplot(fig)

elif mode == "Обучение модели":
    st.header("Обучение модели")
    
    if st.session_state.data is None:
        st.warning("Сначала загрузите данные в разделе 'Загрузка данных'")
    else:
        # Выбор целевой переменной
        target_column = st.selectbox(
            "Выберите целевую переменную",
            st.session_state.data.columns.tolist(),
            index=st.session_state.data.columns.tolist().index('dropout') if 'dropout' in st.session_state.data.columns else 0
        )
        
        # Параметры обучения
        st.subheader("Параметры обучения")
        
        # Выбор модели
        predictor = DropoutPredictor()
        available_models = predictor.get_available_models()
        
        model_name = st.selectbox(
            "Выберите модель",
            list(available_models.keys()),
            index=list(available_models.keys()).index('random_forest')
        )
        
        hyperparameter_tuning = st.checkbox("Выполнить подбор гиперпараметров", False)
        
        # Кнопка для начала обучения
        if st.button("Обучить модель"):
            # Предобработка данных
            with st.spinner("Предобработка данных..."):
                st.session_state.preprocessor, st.session_state.X, st.session_state.y, data_enriched = preprocess_data(
                    st.session_state.data, target_column)
                st.success("Данные успешно преобразованы")
            
            # Обучение модели
            st.session_state.predictor, metrics = train_model(
                st.session_state.X, st.session_state.y, model_name, hyperparameter_tuning)
            
            # Отображение метрик
            st.subheader("Результаты обучения")
            display_metrics(metrics)
            
            # Сохранение метрик в систему мониторинга
            monitor = PerformanceMonitor()
            dataset_info = {
                'size': len(st.session_state.data),
                'target_column': target_column,
                'hyperparameter_tuning': hyperparameter_tuning
            }
            monitor.add_metrics(metrics, model_name, dataset_info)
            st.success("Метрики модели сохранены в системе мониторинга")
            
            # Отображение важности признаков
            if hasattr(st.session_state.preprocessor, 'get_feature_names'):
                feature_names = st.session_state.preprocessor.get_feature_names()
            else:
                feature_names = [f"feature_{i}" for i in range(st.session_state.X.shape[1])]
                
            display_feature_importance(st.session_state.predictor, feature_names)
            
            # Сохранение модели
            save_model = st.checkbox("Сохранить модель", True)
            
            if save_model:
                model_path = st.session_state.predictor.save_model()
                st.success(f"Модель сохранена в {model_path}")
            
            # Анализ рисков отчисления
            analyze_student_risks(st.session_state.predictor, st.session_state.data, st.session_state.X)

elif mode == "Прогнозирование отчисления":
    st.header("Прогнозирование отчисления")
    
    # Выбор: использовать текущую модель или загрузить сохраненную
    model_source = st.radio(
        "Выберите источник модели",
        ["Использовать текущую модель", "Загрузить сохраненную модель"]
    )
    
    if model_source == "Использовать текущую модель":
        if st.session_state.predictor is None:
            st.warning("Сначала обучите модель в разделе 'Обучение модели'")
        else:
            st.success(f"Используется модель: {st.session_state.predictor.model_name}")
            
            # Анализ рисков с текущей моделью
            if st.session_state.X is not None:
                analyze_student_risks(st.session_state.predictor, st.session_state.data, st.session_state.X)
            else:
                st.warning("Данные не подготовлены. Вернитесь к разделу 'Обучение модели'")
    
    else:  # Загрузка сохраненной модели
        # Получаем список доступных моделей
        model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
        
        if not model_files:
            st.warning("Нет сохраненных моделей. Сначала обучите и сохраните модель.")
        else:
            selected_model = st.selectbox("Выберите модель", model_files)
            model_path = os.path.join("models", selected_model)
            
            # Загружаем данные (если не загружены)
            if st.session_state.data is None:
                st.warning("Сначала загрузите данные в разделе 'Загрузка данных'")
            else:
                # Загружаем модель
                if st.button("Загрузить модель и выполнить прогнозирование"):
                    with st.spinner("Загрузка модели..."):
                        predictor = DropoutPredictor()
                        predictor.load_model(model_path)
                        st.session_state.predictor = predictor
                        st.success(f"Модель {predictor.model_name} успешно загружена")
                    
                    # Предобработка данных
                    with st.spinner("Предобработка данных..."):
                        st.session_state.preprocessor, st.session_state.X, st.session_state.y, data_enriched = preprocess_data(
                            st.session_state.data)
                        st.success("Данные успешно преобразованы")
                    
                    # Анализ рисков
                    analyze_student_risks(st.session_state.predictor, st.session_state.data, st.session_state.X)

elif mode == "Мониторинг и анализ":
    st.header("Мониторинг и анализ")
    
    monitoring_tab, risk_tracking_tab = st.tabs(["Мониторинг производительности", "Отслеживание рисков"])
    
    with monitoring_tab:
        st.subheader("Мониторинг производительности моделей")
        
        # Инициализируем монитор
        monitor = PerformanceMonitor()
        
        # Выбор действия
        monitor_action = st.radio(
            "Выберите действие",
            ["Просмотр истории метрик", "Сравнение моделей", "Анализ тренда метрик", "Генерация отчета"]
        )
        
        if monitor_action == "Просмотр истории метрик":
            # Получаем историю метрик
            metrics_history = monitor.get_metrics_history()
            
            if metrics_history.empty:
                st.info("История метрик пуста. Сначала нужно добавить метрики с помощью обучения моделей.")
            else:
                st.dataframe(metrics_history)
                
                # Возможность отфильтровать данные
                st.subheader("Фильтрация метрик")
                
                # Фильтры
                model_versions = metrics_history['model_version'].unique()
                metric_names = metrics_history['metric_name'].unique()
                
                selected_models = st.multiselect("Выберите версии моделей", model_versions)
                selected_metrics = st.multiselect("Выберите метрики", metric_names)
                
                if selected_models and selected_metrics:
                    filtered_history = metrics_history[
                        metrics_history['model_version'].isin(selected_models) & 
                        metrics_history['metric_name'].isin(selected_metrics)
                    ]
                    
                    st.dataframe(filtered_history)
        
        elif monitor_action == "Сравнение моделей":
            # Получаем историю метрик
            metrics_history = monitor.get_metrics_history()
            
            if metrics_history.empty:
                st.info("История метрик пуста. Сначала нужно добавить метрики с помощью обучения моделей.")
            else:
                model_versions = metrics_history['model_version'].unique()
                
                if len(model_versions) < 2:
                    st.warning("Для сравнения нужно как минимум две модели. Обучите еще модели.")
                else:
                    # Выбор моделей для сравнения
                    selected_models = st.multiselect(
                        "Выберите модели для сравнения", 
                        model_versions,
                        default=list(model_versions)[:2]
                    )
                    
                    # Выбор метрик для сравнения
                    metric_names = metrics_history['metric_name'].unique()
                    selected_metrics = st.multiselect(
                        "Выберите метрики для сравнения", 
                        metric_names,
                        default=list(metric_names)[:3]
                    )
                    
                    if selected_models and selected_metrics and len(selected_models) >= 2:
                        with st.spinner("Генерация графика сравнения моделей..."):
                            fig = monitor.compare_models(
                                model_versions=selected_models, 
                                metrics=selected_metrics
                            )
                            st.pyplot(fig)
                    else:
                        st.warning("Выберите минимум две модели и одну метрику")
        
        elif monitor_action == "Анализ тренда метрик":
            # Получаем историю метрик
            metrics_history = monitor.get_metrics_history()
            
            if metrics_history.empty:
                st.info("История метрик пуста. Сначала нужно добавить метрики с помощью обучения моделей.")
            else:
                # Выбор метрики для анализа тренда
                metric_names = metrics_history['metric_name'].unique()
                selected_metric = st.selectbox("Выберите метрику", metric_names)
                
                # Выбор моделей
                model_versions = metrics_history['model_version'].unique()
                selected_models = st.multiselect("Выберите модели", model_versions)
                
                # Размер окна для скользящего среднего
                window_size = st.slider("Размер окна для скользящего среднего", 1, 10, 1)
                
                if selected_metric and selected_models:
                    with st.spinner("Генерация графика тренда метрики..."):
                        fig = monitor.plot_metrics_trend(
                            metric_name=selected_metric,
                            model_versions=selected_models,
                            window=window_size
                        )
                        st.pyplot(fig)
        
        elif monitor_action == "Генерация отчета":
            st.subheader("Генерация отчета о производительности модели")
            
            # Инициализируем генератор отчетов
            report_generator = ReportGenerator()
            
            # Получаем историю метрик
            metrics_history = monitor.get_metrics_history()
            
            if metrics_history.empty:
                st.info("История метрик пуста. Сначала нужно добавить метрики с помощью обучения моделей.")
            else:
                # Выбор модели для отчета
                model_versions = metrics_history['model_version'].unique()
                selected_model = st.selectbox("Выберите модель для отчета", model_versions)
                
                # Фильтруем метрики для выбранной модели
                model_metrics = metrics_history[metrics_history['model_version'] == selected_model]
                
                # Готовим данные для отчета
                metric_names = model_metrics['metric_name'].unique()
                
                if len(metric_names) > 0:
                    # Получаем последнюю оценку рисков для данной модели
                    tracker = StudentRiskTracker()
                    assessment_dates = tracker.get_all_assessment_dates()
                    
                    # Находим оценки, сделанные выбранной моделью
                    model_assessments = []
                    model_assessment_dates = []
                    
                    for date in assessment_dates:
                        assessment = tracker.get_risk_assessment(date)
                        if assessment and assessment['model_version'] == selected_model:
                            model_assessments.append(assessment)
                            model_assessment_dates.append(date)
                    
                    if model_assessments:
                        # Выбираем последнюю оценку
                        selected_date = st.selectbox(
                            "Выберите дату оценки рисков", 
                            model_assessment_dates, 
                            index=len(model_assessment_dates)-1
                        )
                        
                        selected_assessment = tracker.get_risk_assessment(selected_date)
                        
                        # Преобразуем метрики в словарь для отчета
                        latest_metrics = {}
                        for _, row in model_metrics.iterrows():
                            if not isinstance(row['metric_value'], (dict, list)):
                                latest_metrics[row['metric_name']] = row['metric_value']
                        
                        # Генерируем отчет
                        if st.button("Создать отчет"):
                            with st.spinner("Генерация отчета..."):
                                # Добавляем название модели
                                latest_metrics['model_name'] = selected_model
                                
                                # Генерируем отчет
                                report_path = report_generator.generate_risk_report(
                                    student_risks=selected_assessment['student_risks'],
                                    model_info=latest_metrics,
                                    report_title=f"Отчет о рисках отчисления ({selected_date})"
                                )
                                
                                # Открываем отчет для загрузки
                                with open(report_path, 'r', encoding='utf-8') as f:
                                    report_html = f.read()
                                
                                st.success(f"Отчет успешно сгенерирован")
                                st.download_button(
                                    label="Скачать отчет (HTML)",
                                    data=report_html,
                                    file_name=f"risk_report_{selected_date}.html",
                                    mime="text/html"
                                )
                    else:
                        st.warning("Для выбранной модели нет оценок рисков. Сначала выполните прогнозирование с этой моделью.")
                else:
                    st.warning("Нет данных о метриках для выбранной модели.")
                    
    with risk_tracking_tab:
        st.subheader("Отслеживание рисков отчисления студентов")
        
        # Инициализируем трекер рисков
        tracker = StudentRiskTracker()
        
        # Выбор действия
        risk_action = st.radio(
            "Выберите действие",
            ["Сводка по рискам", "Анализ изменений рисков", "Отслеживание студентов"]
        )
        
        # Получаем даты оценок
        assessment_dates = tracker.get_all_assessment_dates()
        
        if not assessment_dates:
            st.info("История рисков пуста. Нужно сначала провести оценку рисков с помощью прогнозирования отчисления.")
        else:
            if risk_action == "Сводка по рискам":
                # Выбор даты для сводки
                selected_date = st.selectbox(
                    "Выберите дату для просмотра сводки", 
                    assessment_dates, 
                    index=len(assessment_dates)-1
                )
                
                # Генерация сводки
                summary = tracker.generate_risk_summary(selected_date)
                
                if summary:
                    st.write(f"### Сводка по рискам на {summary['date']}")
                    st.write(f"**Всего студентов:** {summary['total_students']}")
                    st.write(f"**Средний риск отчисления:** {summary['mean_risk']:.2f}")
                    st.write(f"**Студентов с высоким риском:** {summary['extreme_risk_count']} ({summary['extreme_risk_count']/summary['total_students']*100:.1f}%)")
                    
                    # Распределение по уровням риска
                    risk_df = pd.DataFrame({
                        'Уровень риска': list(summary['risk_levels'].keys()),
                        'Количество студентов': list(summary['risk_levels'].values()),
                        'Процент': [summary['risk_percentages'][k] for k in summary['risk_levels'].keys()]
                    })
                    
                    st.write("#### Распределение по уровням риска")
                    st.dataframe(risk_df)
                    
                    # Визуализация
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['green', 'orange', 'red']
                    ax.pie(
                        risk_df['Процент'], 
                        labels=risk_df['Уровень риска'], 
                        colors=colors,
                        autopct='%1.1f%%', 
                        startangle=90
                    )
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                    # Дополнительная информация
                    st.write(f"**Модель:** {summary['model_version']}")
                    if summary['notes']:
                        st.write(f"**Примечания:** {summary['notes']}")
                
            elif risk_action == "Анализ изменений рисков":
                if len(assessment_dates) < 2:
                    st.warning("Для анализа изменений нужно минимум две оценки рисков.")
                else:
                    # Выбор дат для сравнения
                    col1, col2 = st.columns(2)
                    with col1:
                        date1 = st.selectbox(
                            "Начальная дата", 
                            assessment_dates[:-1],
                            index=0
                        )
                    with col2:
                        date2 = st.selectbox(
                            "Конечная дата", 
                            [d for d in assessment_dates if d > date1],
                            index=0
                        )
                    
                    # Расчет изменений
                    changes = tracker.calculate_risk_changes(date1, date2)
                    
                    if changes:
                        st.write(f"### Изменения рисков с {changes['date_before']} по {changes['date_after']}")
                        st.write(f"**Количество студентов:** {changes['students_count']}")
                        
                        # Изменение среднего риска
                        risk_change = changes['mean_risk_change']
                        change_color = "red" if risk_change > 0 else "green"
                        st.write(f"**Изменение среднего риска:** {risk_change:.3f} ({changes['mean_risk_before']:.3f} → {changes['mean_risk_after']:.3f})")
                        
                        # Статистика изменений
                        st.write("#### Распределение изменений")
                        change_stats_df = pd.DataFrame({
                            'Категория': list(changes['change_stats'].keys()),
                            'Количество студентов': list(changes['change_stats'].values())
                        })
                        st.dataframe(change_stats_df)
                        
                        # Визуализация
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['darkgreen', 'lightgreen', 'gray', 'orange', 'red']
                        ax.bar(
                            change_stats_df['Категория'],
                            change_stats_df['Количество студентов'],
                            color=colors
                        )
                        ax.set_xlabel('Категория изменения')
                        ax.set_ylabel('Количество студентов')
                        ax.set_title('Распределение студентов по категориям изменения рисков')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                        
                        # Студенты с наибольшим улучшением и ухудшением
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### Топ-5 студентов с наибольшим улучшением")
                            improved_df = pd.DataFrame(changes['most_improved'])
                            improved_df['risk_change'] = improved_df['risk_change'].round(3)
                            st.dataframe(improved_df)
                        
                        with col2:
                            st.write("#### Топ-5 студентов с наибольшим ухудшением")
                            deteriorated_df = pd.DataFrame(changes['most_deteriorated'])
                            deteriorated_df['risk_change'] = deteriorated_df['risk_change'].round(3)
                            st.dataframe(deteriorated_df)
                
                # Добавляем возможность создания сравнительного отчета
                if st.button("Создать сравнительный отчет"):
                    with st.spinner("Генерация сравнительного отчета..."):
                        # Получаем данные о рисках
                        assessment1 = tracker.get_risk_assessment(date1)
                        assessment2 = tracker.get_risk_assessment(date2)
                        
                        # Инициализируем генератор отчетов
                        report_generator = ReportGenerator()
                        
                        # Генерируем отчет
                        report_path = report_generator.generate_comparative_report(
                            risk_data1=assessment1['student_risks'],
                            risk_data2=assessment2['student_risks'],
                            date1=date1,
                            date2=date2
                        )
                        
                        # Открываем отчет для загрузки
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_html = f.read()
                        
                        st.success(f"Сравнительный отчет успешно сгенерирован")
                        st.download_button(
                            label="Скачать сравнительный отчет (HTML)",
                            data=report_html,
                            file_name=f"comparative_report_{date1}_vs_{date2}.html",
                            mime="text/html"
                        )
            
            elif risk_action == "Отслеживание студентов":
                # Получаем данные о доступных оценках
                last_assessment_key = assessment_dates[-1]
                last_assessment = tracker.get_risk_assessment(last_assessment_key)
                
                if last_assessment:
                    student_risks_df = last_assessment['student_risks']
                    
                    # Выбор студентов для отслеживания
                    available_students = student_risks_df['student_id'].tolist()
                    selected_students = st.multiselect(
                        "Выберите студентов для отслеживания", 
                        available_students
                    )
                    
                    if selected_students:
                        # Временной диапазон
                        date_range = st.slider(
                            "Выберите период для анализа",
                            min_value=0,
                            max_value=len(assessment_dates)-1,
                            value=(0, len(assessment_dates)-1)
                        )
                        
                        selected_dates = assessment_dates[date_range[0]:date_range[1]+1]
                        
                        # Строим график динамики рисков
                        with st.spinner("Генерация графика динамики рисков..."):
                            fig = tracker.plot_risk_trends(selected_students)
                            st.pyplot(fig)
                            
                        # Детальная информация
                        risk_df = tracker.compare_student_risks(
                            selected_students,
                            start_date=selected_dates[0],
                            end_date=selected_dates[-1]
                        )
                        
                        if not risk_df.empty:
                            st.write("#### Детальная информация о рисках")
                            risk_pivot = risk_df.pivot_table(
                                index='date',
                                columns='student_id',
                                values='dropout_probability'
                            )
                            st.dataframe(risk_pivot)
                            
                            # Возможность экспорта данных
                            csv = risk_df.to_csv(index=False)
                            st.download_button(
                                label="Скачать данные отслеживания",
                                data=csv,
                                file_name='student_risk_tracking.csv',
                                mime='text/csv',
                            )

else:  # О системе
    st.header("О системе")
    
    st.markdown("""
    ## Система прогнозирования отчисления студентов
    
    Интеллектуальный сервис для прогнозирования отчисления студентов Московского университета имени С.Ю. Витте.
    
    ### Возможности системы
    
    - Загрузка и подготовка данных о студентах
    - Обучение моделей машинного обучения для прогнозирования отчисления
    - Визуализация результатов и важности признаков
    - Анализ рисков отчисления
    - Сохранение и загрузка моделей
    
    ### Рекомендации по использованию
    
    1. Загрузите данные о студентах в разделе "Загрузка данных"
    2. Обучите модель в разделе "Обучение модели"
    3. Проанализируйте результаты и риски отчисления
    4. Сохраните модель для дальнейшего использования
    
    ### Требования к данным
    
    Для корректной работы системы рекомендуется использовать данные, содержащие следующие признаки:
    
    - Демографические данные (возраст, пол и т.д.)
    - Академические показатели (средний балл, пропуски занятий и т.д.)  
    - Социально-экономические факторы (наличие стипендии, трудоустройство и т.д.)
    - Личностные характеристики (мотивация, активность и т.д.)
    
    Целевая переменная "dropout" должна иметь значения 0 (не отчислен) или 1 (отчислен).
    """)
    
    # Отображение информации о версии
    st.sidebar.markdown("---")
    st.sidebar.info("Версия: 1.0.0\nДата: Май 2025")

# Footer
st.markdown("---")
st.markdown("Московский университет имени С.Ю. Витте © 2025") 