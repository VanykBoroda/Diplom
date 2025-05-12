import os
import pandas as pd
import numpy as np
import json
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Настройка логирования.
    
    Args:
        log_file: Путь к файлу для сохранения логов
        level: Уровень логирования
    """
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Обработчик для записи в файл, если указан путь
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    logger.info(f"Логирование настроено с уровнем {logging.getLevelName(level)}")


def create_directory_structure(base_dir: str = ".") -> Dict[str, str]:
    """
    Создание структуры директорий для проекта.
    
    Args:
        base_dir: Базовая директория проекта
        
    Returns:
        Dict[str, str]: Словарь с путями к директориям
    """
    # Список необходимых директорий
    directories = {
        'data': os.path.join(base_dir, 'data'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Создаем директории
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Создана директория: {path}")
        
    return directories


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Сохранение конфигурации в JSON-файл.
    
    Args:
        config: Словарь с конфигурацией
        filepath: Путь для сохранения файла
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Конфигурация сохранена в {filepath}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении конфигурации: {str(e)}")
        raise


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из JSON-файла.
    
    Args:
        filepath: Путь к файлу с конфигурацией
        
    Returns:
        Dict[str, Any]: Словарь с конфигурацией
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Конфигурация загружена из {filepath}")
        return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
        raise


def get_experiment_name(model_name: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Генерация уникального имени эксперимента.
    
    Args:
        model_name: Название модели
        params: Параметры модели
        
    Returns:
        str: Уникальное имя эксперимента
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Базовое имя эксперимента
    experiment_name = f"{model_name}_{timestamp}"
    
    # Добавляем ключевые параметры, если они указаны
    if params:
        # Выбираем наиболее важные параметры для имени
        key_params = []
        for key in sorted(params.keys()):
            if key in ['n_estimators', 'max_depth', 'learning_rate', 'C']:
                value = params[key]
                if isinstance(value, (int, float)):
                    key_params.append(f"{key}{value}")
                else:
                    key_params.append(f"{key}{str(value)}")
        
        if key_params:
            experiment_name += "_" + "_".join(key_params)
    
    return experiment_name


def plot_training_history(history: Dict[str, List[float]], 
                        title: str = "Обучение модели", 
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Построение графиков истории обучения модели.
    
    Args:
        history: Словарь с метриками по эпохам
        title: Заголовок графика
        figsize: Размер изображения
        save_path: Путь для сохранения изображения
        
    Returns:
        plt.Figure: Объект Figure с визуализацией
    """
    plt.figure(figsize=figsize)
    
    # Строим графики для каждой метрики
    for i, (metric_name, values) in enumerate(history.items()):
        plt.subplot(1, len(history), i+1)
        plt.plot(values)
        plt.title(metric_name)
        plt.xlabel('Эпоха')
        plt.ylabel('Значение')
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Сохраняем изображение, если указан путь
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"График истории обучения сохранен в {save_path}")
    
    return plt.gcf()


def compare_models(results: List[Dict[str, Any]], 
                metrics: List[str] = ['accuracy', 'f1_score', 'precision', 'recall'],
                figsize: Tuple[int, int] = (14, 8),
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Сравнение нескольких моделей по метрикам.
    
    Args:
        results: Список словарей с результатами оценки моделей
        metrics: Список метрик для сравнения
        figsize: Размер изображения
        save_path: Путь для сохранения изображения
        
    Returns:
        plt.Figure: Объект Figure с визуализацией
    """
    plt.figure(figsize=figsize)
    
    # Преобразуем данные для построения
    model_names = [r['model_name'] for r in results]
    metric_values = {metric: [r.get(metric, 0) for r in results] for metric in metrics}
    
    # Количество метрик и моделей
    n_metrics = len(metrics)
    n_models = len(results)
    
    # Настройка отображения
    bar_width = 0.8 / n_metrics
    index = np.arange(n_models)
    
    # Строим группированную столбчатую диаграмму
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, metric_values[metric], bar_width,
               label=metric, alpha=0.8)
    
    plt.xlabel('Модель')
    plt.ylabel('Значение')
    plt.title('Сравнение моделей по метрикам')
    plt.xticks(index + bar_width * (n_metrics - 1) / 2, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем изображение, если указан путь
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"График сравнения моделей сохранен в {save_path}")
    
    return plt.gcf()


def export_results_to_excel(results: Dict[str, Any], filepath: str) -> None:
    """
    Экспорт результатов анализа в Excel-файл.
    
    Args:
        results: Словарь с результатами
        filepath: Путь для сохранения файла
    """
    # Создаем writer для записи в Excel
    writer = pd.ExcelWriter(filepath, engine='openpyxl')
    
    try:
        # Сохраняем основные метрики
        metrics_df = pd.DataFrame({
            'Метрика': list(results.keys()),
            'Значение': [results[k] for k in results.keys() if not isinstance(results[k], (dict, list))]
        })
        metrics_df.to_excel(writer, sheet_name='Метрики', index=False)
        
        # Сохраняем данные о студентах, если они есть
        if 'student_data' in results and isinstance(results['student_data'], pd.DataFrame):
            results['student_data'].to_excel(writer, sheet_name='Данные студентов', index=False)
        
        # Сохраняем важность признаков, если она есть
        if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
            results['feature_importance'].to_excel(writer, sheet_name='Важность признаков', index=False)
        
        # Сохраняем другие таблицы, если они есть
        for key, value in results.items():
            if isinstance(value, pd.DataFrame) and key not in ['student_data', 'feature_importance']:
                value.to_excel(writer, sheet_name=key[:31], index=False)  # Ограничение длины имени листа
        
        writer.save()
        logger.info(f"Результаты экспортированы в {filepath}")
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте результатов: {str(e)}")
        raise
    finally:
        writer.close()


def calculate_model_performance_stats(evaluations: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Расчет статистики производительности моделей на основе нескольких оценок.
    
    Args:
        evaluations: Список словарей с результатами оценки
        
    Returns:
        Dict[str, Dict[str, float]]: Словарь со статистикой по метрикам
    """
    # Метрики для анализа
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Получаем все уникальные модели
    model_names = set(eval_data['model_name'] for eval_data in evaluations)
    
    # Создаем словарь для хранения статистики
    stats = {}
    
    for model_name in model_names:
        # Фильтруем оценки для текущей модели
        model_evals = [e for e in evaluations if e['model_name'] == model_name]
        
        # Словарь для хранения статистики по метрикам
        model_stats = {}
        
        for metric in metrics:
            values = [e.get(metric, 0) for e in model_evals]
            if values:
                model_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        stats[model_name] = model_stats
    
    return stats


if __name__ == "__main__":
    # Пример использования
    
    # Настройка логирования
    setup_logging("logs/utils_test.log")
    
    # Создание структуры директорий
    dirs = create_directory_structure()
    
    # Сохранение тестовой конфигурации
    config = {
        'model': 'random_forest',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'data': {
            'test_size': 0.2,
            'random_state': 42
        }
    }
    
    config_path = os.path.join(dirs['models'], 'test_config.json')
    save_config(config, config_path)
    
    # Загрузка конфигурации
    loaded_config = load_config(config_path)
    print("Загруженная конфигурация:", loaded_config)
    
    # Генерация имени эксперимента
    experiment_name = get_experiment_name('random_forest', config['hyperparameters'])
    print("Имя эксперимента:", experiment_name) 