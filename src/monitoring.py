import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Класс для мониторинга производительности моделей и отслеживания изменений во времени.
    Позволяет отслеживать дрейф в производительности модели и сравнивать разные версии моделей.
    """
    
    def __init__(self, monitor_dir: str = "monitoring"):
        """
        Инициализация системы мониторинга.
        
        Args:
            monitor_dir: Директория для хранения данных мониторинга
        """
        self.monitor_dir = monitor_dir
        os.makedirs(monitor_dir, exist_ok=True)
        
        # Путь к файлу с историей метрик
        self.metrics_history_path = os.path.join(monitor_dir, "metrics_history.json")
        
        # Загружаем историю метрик, если файл существует
        if os.path.exists(self.metrics_history_path):
            with open(self.metrics_history_path, 'r', encoding='utf-8') as f:
                self.metrics_history = json.load(f)
        else:
            self.metrics_history = []
    
    def add_metrics(self, metrics: Dict[str, Any], model_version: str, 
                   dataset_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Добавление новых метрик в историю.
        
        Args:
            metrics: Словарь с метриками модели
            model_version: Версия или имя модели
            dataset_info: Информация о наборе данных (опционально)
        """
        # Создаем запись с метриками и метаданными
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'metrics': metrics
        }
        
        # Добавляем информацию о наборе данных, если она предоставлена
        if dataset_info:
            entry['dataset_info'] = dataset_info
            
        # Добавляем запись в историю
        self.metrics_history.append(entry)
        
        # Сохраняем обновленную историю
        self._save_metrics_history()
        
        logger.info(f"Добавлены новые метрики для модели {model_version}")
    
    def _save_metrics_history(self) -> None:
        """
        Сохранение истории метрик в файл.
        """
        with open(self.metrics_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=4, ensure_ascii=False)
            
        logger.info(f"История метрик сохранена в {self.metrics_history_path}")
    
    def get_metrics_history(self, model_version: Optional[str] = None, 
                          metric_name: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Получение истории метрик с возможностью фильтрации.
        
        Args:
            model_version: Фильтр по версии модели
            metric_name: Фильтр по названию метрики
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame с историей метрик
        """
        # Преобразуем историю в DataFrame
        history_data = []
        
        for entry in self.metrics_history:
            entry_timestamp = datetime.fromisoformat(entry['timestamp'])
            entry_model_version = entry['model_version']
            
            # Проверяем фильтр по версии модели
            if model_version and entry_model_version != model_version:
                continue
                
            # Проверяем фильтр по датам
            if start_date:
                start_datetime = datetime.fromisoformat(start_date)
                if entry_timestamp < start_datetime:
                    continue
                    
            if end_date:
                end_datetime = datetime.fromisoformat(end_date)
                if entry_timestamp > end_datetime:
                    continue
            
            # Обрабатываем метрики
            for metric_key, metric_value in entry['metrics'].items():
                # Пропускаем сложные структуры вроде матрицы ошибок
                if isinstance(metric_value, dict) or isinstance(metric_value, list):
                    continue
                    
                # Проверяем фильтр по названию метрики
                if metric_name and metric_key != metric_name:
                    continue
                    
                # Добавляем запись
                history_data.append({
                    'timestamp': entry_timestamp,
                    'model_version': entry_model_version,
                    'metric_name': metric_key,
                    'metric_value': metric_value,
                    'dataset_info': entry.get('dataset_info', {})
                })
        
        # Создаем DataFrame
        df = pd.DataFrame(history_data)
        
        return df
    
    def plot_metrics_trend(self, metric_name: str, model_versions: Optional[List[str]] = None,
                          window: int = 1, figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика динамики метрики во времени.
        
        Args:
            metric_name: Название метрики для отображения
            model_versions: Список версий моделей для сравнения (если None, все версии)
            window: Размер окна для скользящего среднего
            figsize: Размер изображения
            save_path: Путь для сохранения изображения
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        # Получаем историю метрик
        history_df = self.get_metrics_history(metric_name=metric_name)
        
        if history_df.empty:
            logger.warning(f"Нет данных для метрики {metric_name}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Нет данных для метрики {metric_name}", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Фильтруем по версиям моделей, если указаны
        if model_versions:
            history_df = history_df[history_df['model_version'].isin(model_versions)]
            
        # Сортируем по времени
        history_df = history_df.sort_values('timestamp')
        
        # Создаем график
        fig, ax = plt.subplots(figsize=figsize)
        
        # Строим линии для каждой версии модели
        for model_version, group in history_df.groupby('model_version'):
            # Применяем скользящее среднее, если window > 1
            if window > 1 and len(group) >= window:
                values = group['metric_value'].rolling(window=window).mean()
            else:
                values = group['metric_value']
                
            ax.plot(group['timestamp'], values, marker='o', linewidth=2, label=model_version)
            
        ax.set_xlabel('Дата')
        ax.set_ylabel(f'Значение метрики {metric_name}')
        ax.set_title(f'Динамика метрики {metric_name} во времени')
        ax.legend()
        ax.grid(True)
        
        # Форматируем ось X для лучшего отображения дат
        fig.autofmt_xdate()
        
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"График динамики метрики сохранен в {save_path}")
            
        return fig
    
    def compare_models(self, model_versions: List[str], 
                     metrics: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (14, 8),
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Сравнение производительности нескольких версий моделей.
        
        Args:
            model_versions: Список версий моделей для сравнения
            metrics: Список метрик для сравнения (если None, используются все доступные)
            figsize: Размер изображения
            save_path: Путь для сохранения изображения
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        # Получаем историю метрик
        history_df = self.get_metrics_history()
        
        # Фильтруем по указанным версиям моделей
        history_df = history_df[history_df['model_version'].isin(model_versions)]
        
        if history_df.empty:
            logger.warning("Нет данных для указанных версий моделей")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Нет данных для указанных версий моделей", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Если метрики не указаны, используем все доступные
        if metrics is None:
            metrics = history_df['metric_name'].unique().tolist()
        else:
            # Фильтруем только по указанным метрикам
            history_df = history_df[history_df['metric_name'].isin(metrics)]
            
        # Вычисляем среднее значение метрики для каждой версии модели
        agg_df = history_df.groupby(['model_version', 'metric_name'])['metric_value'].mean().reset_index()
        
        # Преобразуем данные для построения графика
        pivot_df = agg_df.pivot(index='model_version', columns='metric_name', values='metric_value')
        
        # Создаем график
        fig, ax = plt.subplots(figsize=figsize)
        
        # Строим столбчатую диаграмму
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_xlabel('Версия модели')
        ax.set_ylabel('Значение метрики')
        ax.set_title('Сравнение производительности моделей')
        ax.legend(title='Метрика')
        ax.grid(True, axis='y')
        
        # Добавляем значения на вершины столбцов
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
            
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"График сравнения моделей сохранен в {save_path}")
            
        return fig


class StudentRiskTracker:
    """
    Класс для отслеживания и анализа рисков отчисления студентов во времени.
    Позволяет выявлять тенденции и отслеживать эффективность мер по снижению рисков.
    """
    
    def __init__(self, storage_dir: str = "risk_tracking"):
        """
        Инициализация трекера рисков.
        
        Args:
            storage_dir: Директория для хранения данных об отслеживании рисков
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Путь к файлу с историей рисков
        self.risk_history_path = os.path.join(storage_dir, "risk_history.pkl")
        
        # Загружаем историю рисков, если файл существует
        if os.path.exists(self.risk_history_path):
            with open(self.risk_history_path, 'rb') as f:
                self.risk_history = pickle.load(f)
        else:
            self.risk_history = {}
    
    def add_risk_assessment(self, assessment_date: datetime, 
                          student_risks: pd.DataFrame,
                          model_version: str,
                          notes: Optional[str] = None) -> None:
        """
        Добавление новой оценки рисков в историю.
        
        Args:
            assessment_date: Дата проведения оценки рисков
            student_risks: DataFrame с оценками риска для студентов
            model_version: Версия модели, использованной для оценки
            notes: Дополнительные примечания
        """
        # Создаем ключ для даты
        date_key = assessment_date.strftime("%Y-%m-%d")
        
        # Сохраняем данные оценки
        assessment_data = {
            'date': assessment_date,
            'student_risks': student_risks.copy(),
            'model_version': model_version,
            'notes': notes
        }
        
        # Добавляем в историю
        self.risk_history[date_key] = assessment_data
        
        # Сохраняем обновленную историю
        self._save_risk_history()
        
        logger.info(f"Добавлена новая оценка рисков от {date_key}")
    
    def _save_risk_history(self) -> None:
        """
        Сохранение истории рисков в файл.
        """
        with open(self.risk_history_path, 'wb') as f:
            pickle.dump(self.risk_history, f)
            
        logger.info(f"История рисков сохранена в {self.risk_history_path}")
    
    def get_risk_assessment(self, date_key: str) -> Optional[Dict[str, Any]]:
        """
        Получение оценки рисков для указанной даты.
        
        Args:
            date_key: Ключ даты в формате 'YYYY-MM-DD'
            
        Returns:
            Optional[Dict[str, Any]]: Данные оценки рисков или None, если не найдено
        """
        return self.risk_history.get(date_key)
    
    def get_all_assessment_dates(self) -> List[str]:
        """
        Получение списка всех дат, для которых есть оценки рисков.
        
        Returns:
            List[str]: Список дат в формате 'YYYY-MM-DD'
        """
        return sorted(list(self.risk_history.keys()))
    
    def compare_student_risks(self, student_ids: Union[str, List[str]], 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Сравнение рисков для конкретных студентов за период времени.
        
        Args:
            student_ids: ID студента или список ID студентов
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame с историей рисков для указанных студентов
        """
        # Преобразуем одиночный ID в список
        if isinstance(student_ids, str):
            student_ids = [student_ids]
            
        # Фильтруем даты по периоду
        all_dates = self.get_all_assessment_dates()
        filtered_dates = []
        
        for date_key in all_dates:
            if start_date and date_key < start_date:
                continue
            if end_date and date_key > end_date:
                continue
            filtered_dates.append(date_key)
            
        if not filtered_dates:
            logger.warning("Нет данных для указанного периода")
            return pd.DataFrame()
            
        # Собираем историю рисков
        risk_data = []
        
        for date_key in filtered_dates:
            assessment = self.risk_history[date_key]
            student_risks_df = assessment['student_risks']
            
            # Фильтруем только интересующих студентов
            for student_id in student_ids:
                student_row = student_risks_df[student_risks_df['student_id'] == student_id]
                
                if not student_row.empty:
                    risk_prob = student_row['dropout_probability'].values[0]
                    risk_level = student_row['risk_level'].values[0]
                    
                    risk_data.append({
                        'date': date_key,
                        'student_id': student_id,
                        'dropout_probability': risk_prob,
                        'risk_level': risk_level,
                        'model_version': assessment['model_version']
                    })
        
        # Создаем DataFrame
        risk_df = pd.DataFrame(risk_data)
        
        return risk_df
    
    def plot_risk_trends(self, student_ids: Union[str, List[str]], 
                       figsize: Tuple[int, int] = (12, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика изменения рисков для студентов во времени.
        
        Args:
            student_ids: ID студента или список ID студентов
            figsize: Размер изображения
            save_path: Путь для сохранения изображения
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        # Получаем историю рисков
        risk_df = self.compare_student_risks(student_ids)
        
        if risk_df.empty:
            logger.warning("Нет данных для указанных студентов")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Нет данных для указанных студентов", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
            
        # Преобразуем даты
        risk_df['date'] = pd.to_datetime(risk_df['date'])
        
        # Создаем график
        fig, ax = plt.subplots(figsize=figsize)
        
        # Строим линии для каждого студента
        for student_id, group in risk_df.groupby('student_id'):
            group = group.sort_values('date')
            ax.plot(group['date'], group['dropout_probability'], marker='o', linewidth=2, label=student_id)
            
        # Добавляем горизонтальные линии для уровней риска
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Низкий риск')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Средний риск')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Высокий риск')
        
        ax.set_xlabel('Дата')
        ax.set_ylabel('Вероятность отчисления')
        ax.set_title('Динамика риска отчисления во времени')
        ax.legend()
        ax.grid(True)
        
        # Устанавливаем пределы оси Y
        ax.set_ylim(0, 1)
        
        # Форматируем ось X для лучшего отображения дат
        fig.autofmt_xdate()
        
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"График динамики рисков сохранен в {save_path}")
            
        return fig
    
    def generate_risk_summary(self, date_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерация сводки по рискам отчисления.
        
        Args:
            date_key: Ключ даты в формате 'YYYY-MM-DD' (если None, используется последняя доступная дата)
            
        Returns:
            Dict[str, Any]: Словарь со сводной информацией о рисках
        """
        # Если дата не указана, используем последнюю доступную
        if date_key is None:
            all_dates = self.get_all_assessment_dates()
            if not all_dates:
                logger.warning("Нет доступных оценок рисков")
                return {}
            date_key = all_dates[-1]
            
        # Получаем оценку рисков
        assessment = self.get_risk_assessment(date_key)
        if not assessment:
            logger.warning(f"Оценка рисков для даты {date_key} не найдена")
            return {}
            
        student_risks_df = assessment['student_risks']
        
        # Рассчитываем статистику
        risk_levels = student_risks_df['risk_level'].value_counts().to_dict()
        total_students = len(student_risks_df)
        
        # Рассчитываем процентное соотношение
        risk_percentages = {k: (v / total_students * 100) for k, v in risk_levels.items()}
        
        # Рассчитываем средний риск
        mean_risk = student_risks_df['dropout_probability'].mean()
        
        # Находим студентов с экстремально высоким риском
        high_risk_threshold = 0.8
        extreme_risk_students = student_risks_df[student_risks_df['dropout_probability'] >= high_risk_threshold]
        
        # Готовим сводку
        summary = {
            'date': date_key,
            'total_students': total_students,
            'risk_levels': risk_levels,
            'risk_percentages': risk_percentages,
            'mean_risk': mean_risk,
            'extreme_risk_count': len(extreme_risk_students),
            'model_version': assessment['model_version'],
            'notes': assessment.get('notes')
        }
        
        return summary
    
    def calculate_risk_changes(self, date_key1: str, date_key2: str) -> Dict[str, Any]:
        """
        Расчет изменений в рисках между двумя датами.
        
        Args:
            date_key1: Первая дата в формате 'YYYY-MM-DD'
            date_key2: Вторая дата в формате 'YYYY-MM-DD'
            
        Returns:
            Dict[str, Any]: Словарь с информацией об изменениях
        """
        # Получаем оценки рисков
        assessment1 = self.get_risk_assessment(date_key1)
        assessment2 = self.get_risk_assessment(date_key2)
        
        if not assessment1 or not assessment2:
            logger.warning(f"Одна из оценок рисков не найдена")
            return {}
            
        df1 = assessment1['student_risks']
        df2 = assessment2['student_risks']
        
        # Объединяем данные по ID студентов
        merged_df = pd.merge(
            df1[['student_id', 'dropout_probability', 'risk_level']], 
            df2[['student_id', 'dropout_probability', 'risk_level']], 
            on='student_id', 
            suffixes=('_before', '_after')
        )
        
        # Рассчитываем изменения
        merged_df['risk_change'] = merged_df['dropout_probability_after'] - merged_df['dropout_probability_before']
        
        # Классифицируем изменения
        merged_df['change_category'] = pd.cut(
            merged_df['risk_change'],
            bins=[-1, -0.2, -0.05, 0.05, 0.2, 1],
            labels=['Значительное улучшение', 'Улучшение', 'Без изменений', 'Ухудшение', 'Значительное ухудшение']
        )
        
        # Собираем статистику изменений
        change_stats = merged_df['change_category'].value_counts().to_dict()
        
        # Рассчитываем изменение среднего риска
        mean_before = df1['dropout_probability'].mean()
        mean_after = df2['dropout_probability'].mean()
        mean_change = mean_after - mean_before
        
        # Находим студентов с наибольшим улучшением и ухудшением
        merged_df = merged_df.sort_values('risk_change')
        most_improved = merged_df.head(5)[['student_id', 'risk_change']].to_dict('records')
        most_deteriorated = merged_df.tail(5)[['student_id', 'risk_change']].to_dict('records')
        
        # Готовим отчет об изменениях
        changes = {
            'date_before': date_key1,
            'date_after': date_key2,
            'students_count': len(merged_df),
            'change_stats': change_stats,
            'mean_risk_before': mean_before,
            'mean_risk_after': mean_after,
            'mean_risk_change': mean_change,
            'most_improved': most_improved,
            'most_deteriorated': most_deteriorated,
            'model_version_before': assessment1['model_version'],
            'model_version_after': assessment2['model_version']
        }
        
        return changes


if __name__ == "__main__":
    # Пример использования
    
    # Инициализация мониторинга производительности
    monitor = PerformanceMonitor()
    
    # Добавление метрик для разных версий моделей
    metrics1 = {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.79, 'f1': 0.80}
    monitor.add_metrics(metrics1, model_version="random_forest_v1")
    
    metrics2 = {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.81, 'f1': 0.82}
    monitor.add_metrics(metrics2, model_version="random_forest_v2")
    
    # Инициализация трекера рисков студентов
    tracker = StudentRiskTracker()
    
    # Создание примера данных о рисках
    student_risks = pd.DataFrame({
        'student_id': [f'S{i:04d}' for i in range(10)],
        'dropout_probability': np.random.uniform(0, 1, 10),
        'prediction': np.random.randint(0, 2, 10),
        'risk_level': np.random.choice(['Низкий', 'Средний', 'Высокий'], 10)
    })
    
    # Добавление оценки рисков
    today = datetime.now()
    tracker.add_risk_assessment(today, student_risks, model_version="random_forest_v1")
    
    # Генерация сводки по рискам
    summary = tracker.generate_risk_summary()
    print(f"Сводка по рискам на {today.strftime('%Y-%m-%d')}:")
    print(f"Всего студентов: {summary['total_students']}")
    print(f"Распределение по уровням риска: {summary['risk_levels']}") 