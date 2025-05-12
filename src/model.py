import numpy as np
import pandas as pd
import os
import joblib
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DropoutPredictor:
    """
    Класс для прогнозирования отчисления студентов с использованием 
    различных моделей машинного обучения.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Инициализация предиктора отчислений.
        
        Args:
            models_dir: Директория для сохранения/загрузки моделей
        """
        self.models_dir = models_dir
        self.model = None
        self.model_name = None
        self.training_history = {}
        
        # Создаем директорию для моделей, если она не существует
        os.makedirs(models_dir, exist_ok=True)
        
    def get_available_models(self) -> Dict[str, BaseEstimator]:
        """
        Получение списка доступных моделей.
        
        Returns:
            Dict[str, BaseEstimator]: Словарь с названиями и объектами моделей
        """
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'adaboost': AdaBoostClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'neural_network': MLPClassifier(max_iter=1000, random_state=42)
        }
        
        # Создаем ансамбль моделей (Stacking)
        base_models = [
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        
        models['stacking'] = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42)
        )
        
        return models
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              model_name: str = 'random_forest',
              test_size: float = 0.2, 
              hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Обучение модели для прогнозирования отчисления.
        
        Args:
            X: Обработанные признаки
            y: Целевая переменная (отчислен или нет)
            model_name: Название модели для обучения
            test_size: Доля данных для тестирования
            hyperparameter_tuning: Выполнять ли подбор гиперпараметров
            
        Returns:
            Dict[str, Any]: Словарь с метриками качества модели
        """
        logger.info(f"Начало обучения модели {model_name}")
        start_time = time.time()
        
        # Проверяем, существует ли указанная модель
        available_models = self.get_available_models()
        if model_name not in available_models:
            raise ValueError(f"Модель {model_name} не найдена. Доступные модели: {list(available_models.keys())}")
            
        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Выбираем модель
        self.model_name = model_name
        self.model = available_models[model_name]
        
        # Подбор гиперпараметров, если включен
        if hyperparameter_tuning:
            logger.info("Начало подбора гиперпараметров")
            self.model = self._perform_hyperparameter_tuning(
                self.model, X_train, y_train, model_name
            )
            
        # Обучаем модель
        self.model.fit(X_train, y_train)
        
        # Оцениваем модель на тестовой выборке
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Рассчитываем метрики
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        
        # Выполняем кросс-валидацию
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        # Сохраняем историю обучения
        training_time = time.time() - start_time
        
        self.training_history = {
            'model_name': model_name,
            'training_time': training_time,
            'metrics': metrics,
            'hyperparameter_tuning': hyperparameter_tuning,
        }
        
        logger.info(f"Обучение модели {model_name} завершено за {training_time:.2f} сек.")
        logger.info(f"Метрики: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Прогнозирование отчисления студентов.
        
        Args:
            X: Обработанные признаки
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Кортеж (predictions, probabilities)
                с предсказаниями и вероятностями
        """
        if self.model is None:
            raise ValueError("Модель не обучена или не загружена")
            
        logger.info("Выполнение прогнозирования")
        
        # Предсказываем классы
        predictions = self.model.predict(X)
        
        # Предсказываем вероятности, если метод доступен
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
            
        logger.info(f"Прогнозирование выполнено для {len(predictions)} экземпляров")
        return predictions, probabilities
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Сохранение обученной модели в файл.
        
        Args:
            filename: Имя файла для сохранения (если None, генерируется автоматически)
            
        Returns:
            str: Путь к сохраненной модели
        """
        if self.model is None:
            raise ValueError("Нет обученной модели для сохранения")
            
        # Генерируем имя файла, если не указано
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.joblib"
            
        # Полный путь к файлу
        filepath = os.path.join(self.models_dir, filename)
        
        # Сохраняем модель
        data_to_save = {
            'model': self.model,
            'model_name': self.model_name,
            'training_history': self.training_history
        }
        
        joblib.dump(data_to_save, filepath)
        logger.info(f"Модель сохранена в файл {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Загрузка модели из файла.
        
        Args:
            filepath: Путь к файлу с сохраненной моделью
        """
        logger.info(f"Загрузка модели из файла {filepath}")
        
        try:
            # Загружаем данные
            data = joblib.load(filepath)
            
            # Проверяем структуру данных
            if not isinstance(data, dict) or 'model' not in data:
                # Для обратной совместимости - если сохранена только модель
                self.model = data
                self.model_name = os.path.basename(filepath).split('.')[0]
            else:
                # Загружаем модель и метаданные
                self.model = data['model']
                self.model_name = data['model_name']
                self.training_history = data.get('training_history', {})
                
            logger.info(f"Модель {self.model_name} успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
    
    def _perform_hyperparameter_tuning(self, model: BaseEstimator, 
                                      X_train: np.ndarray, y_train: np.ndarray, 
                                      model_name: str) -> BaseEstimator:
        """
        Подбор гиперпараметров модели с использованием GridSearchCV.
        
        Args:
            model: Исходная модель
            X_train: Обучающие данные
            y_train: Целевые значения
            model_name: Название модели
            
        Returns:
            BaseEstimator: Модель с оптимальными гиперпараметрами
        """
        # Словарь с гиперпараметрами для разных моделей
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'adaboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        # Используем параметры по умолчанию для Stacking и других моделей,
        # для которых нет определенных параметров
        if model_name not in param_grids:
            logger.info(f"Подбор гиперпараметров не выполняется для модели {model_name}")
            return model
            
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Лучшие параметры для {model_name}: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Расчет метрик качества модели.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            y_prob: Предсказанные вероятности (если доступны)
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Добавляем AUC-ROC, если доступны вероятности
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
        return metrics
    
    def get_feature_importances(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Получение важности признаков для моделей, которые поддерживают эту функциональность.
        
        Args:
            feature_names: Список названий признаков
            
        Returns:
            Optional[pd.DataFrame]: DataFrame с важностью признаков или None
        """
        if self.model is None:
            raise ValueError("Модель не обучена или не загружена")
            
        # Проверяем, поддерживает ли модель получение важности признаков
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Создаем DataFrame с важностью признаков
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            
            return importance_df
            
        # Для линейных моделей можно использовать коэффициенты
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            indices = np.argsort(np.abs(coef))[::-1]
            
            # Создаем DataFrame с коэффициентами
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'coefficient': coef[indices]
            })
            
            return importance_df
            
        else:
            logger.warning(f"Модель {self.model_name} не поддерживает получение важности признаков")
            return None


if __name__ == "__main__":
    # Пример использования
    from data_loader import DataLoader
    from preprocessor import StudentDataPreprocessor
    
    # Создаем синтетические данные
    loader = DataLoader()
    data = loader.generate_sample_data(2000)
    
    # Предобрабатываем данные
    preprocessor = StudentDataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    
    # Обучаем модель
    predictor = DropoutPredictor()
    metrics = predictor.train(X, y, model_name='random_forest')
    
    # Выводим метрики
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Сохраняем модель
    model_path = predictor.save_model()
    
    # Загружаем модель
    new_predictor = DropoutPredictor()
    new_predictor.load_model(model_path)
    
    # Делаем предсказания
    predictions, probabilities = new_predictor.predict(X[:5])
    print("Предсказания:", predictions)
    if probabilities is not None:
        print("Вероятности:", probabilities) 