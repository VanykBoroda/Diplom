import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    precision_recall_curve, auc, average_precision_score
)

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Класс для оценки моделей и анализа результатов прогнозирования отчисления студентов.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Инициализация оценщика моделей.
        
        Args:
            results_dir: Директория для сохранения результатов
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: Optional[np.ndarray] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Оценка модели с использованием различных метрик.
        
        Args:
            y_true: Истинные значения целевой переменной
            y_pred: Предсказанные значения
            y_prob: Предсказанные вероятности (для ROC-кривой)
            model_name: Название модели
            
        Returns:
            Dict[str, Any]: Словарь с метриками и результатами
        """
        logger.info(f"Оценка модели {model_name}")
        
        # Получаем отчет классификации
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        
        # Рассчитываем матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)
        
        # Собираем результаты в словарь
        evaluation_results = {
            'model_name': model_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'classification_report': report_dict,
            'confusion_matrix': cm.tolist(),
            'accuracy': report_dict['accuracy'],
            'precision': report_dict['1']['precision'] if '1' in report_dict else report_dict['1.0']['precision'],
            'recall': report_dict['1']['recall'] if '1' in report_dict else report_dict['1.0']['recall'],
            'f1_score': report_dict['1']['f1-score'] if '1' in report_dict else report_dict['1.0']['f1-score'],
        }
        
        # Добавляем метрики, связанные с вероятностями, если они доступны
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            
            evaluation_results.update({
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc
                },
                'pr_curve': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'auc': pr_auc
                }
            })
        
        logger.info(f"Оценка завершена. Accuracy: {evaluation_results['accuracy']:.4f}, "
                   f"F1: {evaluation_results['f1_score']:.4f}")
        
        return evaluation_results
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                              filename: Optional[str] = None) -> str:
        """
        Сохранение результатов оценки в файл.
        
        Args:
            results: Словарь с результатами оценки
            filename: Имя файла для сохранения (если None, генерируется автоматически)
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = results.get('model_name', 'model')
            filename = f"{model_name}_evaluation_{timestamp}.json"
            
        filepath = os.path.join(self.results_dir, filename)
        
        # Преобразуем numpy типы в родные типы Python для сериализации в JSON
        results_json = self._convert_numpy_types(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Результаты оценки сохранены в {filepath}")
        return filepath
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Преобразование numpy типов в родные типы Python для сериализации в JSON.
        
        Args:
            obj: Объект для преобразования
            
        Returns:
            Any: Преобразованный объект
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return obj
        
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str] = ["Не отчислен", "Отчислен"],
                            figure_size: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение матрицы ошибок.
        
        Args:
            confusion_matrix: Матрица ошибок
            class_names: Названия классов
            figure_size: Размер изображения
            save_path: Путь для сохранения изображения (если None, не сохраняется)
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        plt.figure(figsize=figure_size)
        
        # Нормализуем матрицу по строкам
        cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Создаем тепловую карту
        ax = sns.heatmap(cm_norm, annot=confusion_matrix, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
        
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.title('Матрица ошибок')
        
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Матрица ошибок сохранена в {save_path}")
            
        return plt.gcf()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                     figure_size: Tuple[int, int] = (8, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение ROC-кривой.
        
        Args:
            fpr: False Positive Rate
            tpr: True Positive Rate
            roc_auc: Площадь под ROC-кривой
            figure_size: Размер изображения
            save_path: Путь для сохранения изображения (если None, не сохраняется)
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        plt.figure(figsize=figure_size)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"ROC-кривая сохранена в {save_path}")
            
        return plt.gcf()
    
    def plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray, 
                                  pr_auc: float, figure_size: Tuple[int, int] = (8, 6),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение кривой точности-полноты.
        
        Args:
            precision: Точность
            recall: Полнота
            pr_auc: Площадь под кривой точности-полноты
            figure_size: Размер изображения
            save_path: Путь для сохранения изображения (если None, не сохраняется)
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        plt.figure(figsize=figure_size)
        
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (area = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Кривая точности-полноты сохранена в {save_path}")
            
        return plt.gcf()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                             top_n: int = 10, figure_size: Tuple[int, int] = (10, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика важности признаков.
        
        Args:
            feature_importance: DataFrame с важностью признаков
            top_n: Количество лучших признаков для отображения
            figure_size: Размер изображения
            save_path: Путь для сохранения изображения (если None, не сохраняется)
            
        Returns:
            plt.Figure: Объект Figure с визуализацией
        """
        # Выбираем top_n лучших признаков
        if len(feature_importance) > top_n:
            feature_importance = feature_importance.head(top_n)
            
        plt.figure(figsize=figure_size)
        
        # Выбираем столбец с важностью, в зависимости от модели
        importance_col = 'importance' if 'importance' in feature_importance.columns else 'coefficient'
        
        # Сортируем признаки и строим горизонтальную столбчатую диаграмму
        ax = sns.barplot(
            x=feature_importance[importance_col], 
            y=feature_importance['feature'],
            palette='viridis'
        )
        
        plt.title(f'Топ-{top_n} важных признаков')
        plt.xlabel('Важность')
        plt.ylabel('Признак')
        plt.tight_layout()
        
        # Сохраняем изображение, если указан путь
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"График важности признаков сохранен в {save_path}")
            
        return plt.gcf()
    
    def analyze_student_risks(self, X_student: pd.DataFrame, dropout_probs: np.ndarray, 
                            feature_importance: Optional[pd.DataFrame] = None,
                            threshold: float = 0.5) -> pd.DataFrame:
        """
        Анализ риска отчисления студентов с выявлением ключевых факторов.
        
        Args:
            X_student: DataFrame с данными студентов
            dropout_probs: Вероятности отчисления
            feature_importance: DataFrame с важностью признаков (если доступно)
            threshold: Порог для определения риска отчисления
            
        Returns:
            pd.DataFrame: DataFrame с результатами анализа
        """
        logger.info("Анализ рисков отчисления студентов")
        
        # Создаем DataFrame с результатами
        results = pd.DataFrame({
            'student_id': X_student.index if isinstance(X_student, pd.DataFrame) else np.arange(len(dropout_probs)),
            'dropout_probability': dropout_probs,
            'risk_level': pd.cut(
                dropout_probs, 
                bins=[0, 0.3, 0.6, 1.0], 
                labels=['Низкий', 'Средний', 'Высокий']
            )
        })
        
        # Если доступна важность признаков, анализируем ключевые факторы для каждого студента
        if feature_importance is not None and isinstance(X_student, pd.DataFrame):
            # Получаем наиболее важные признаки
            top_features = feature_importance['feature'].tolist()[:10]
            
            # Анализируем каждого студента с высоким риском
            high_risk_students = results[results['dropout_probability'] >= threshold]
            
            for idx in high_risk_students.index:
                student_data = X_student.loc[results.loc[idx, 'student_id']]
                
                # Находим ключевые факторы для этого студента
                key_factors = []
                for feature in top_features:
                    if feature in student_data.index:
                        value = student_data[feature]
                        # Для категориальных признаков после OneHotEncoding
                        if feature.endswith('_1') and value == 1:
                            key_factors.append(feature.replace('_1', ''))
                        # Для числовых признаков сравниваем с медианой
                        elif not feature.endswith(('_0', '_1')) and value > X_student[feature].median():
                            key_factors.append(f"{feature} ({value:.2f})")
                            
                # Сохраняем ключевые факторы
                results.loc[idx, 'key_factors'] = ', '.join(key_factors[:3])  # Топ-3 фактора
        
        logger.info(f"Найдено {len(results[results['risk_level'] == 'Высокий'])} студентов с высоким риском отчисления")
        return results


if __name__ == "__main__":
    # Пример использования
    from data_loader import DataLoader
    from preprocessor import StudentDataPreprocessor
    from model import DropoutPredictor
    import os
    
    # Генерируем тестовые данные
    loader = DataLoader()
    data = loader.generate_sample_data(1000)
    
    # Предобрабатываем данные
    preprocessor = StudentDataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    
    # Обучаем модель
    predictor = DropoutPredictor()
    metrics = predictor.train(X, y, model_name='random_forest')
    
    # Делаем предсказания
    y_pred, y_prob = predictor.predict(X)
    
    # Оцениваем модель
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_model(y, y_pred, y_prob, model_name='random_forest')
    
    # Сохраняем результаты
    results_path = evaluator.save_evaluation_results(eval_results)
    
    # Строим визуализации
    cm_path = os.path.join(evaluator.results_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(np.array(eval_results['confusion_matrix']), save_path=cm_path)
    
    if 'roc_curve' in eval_results:
        roc_path = os.path.join(evaluator.results_dir, 'roc_curve.png')
        evaluator.plot_roc_curve(
            np.array(eval_results['roc_curve']['fpr']),
            np.array(eval_results['roc_curve']['tpr']),
            eval_results['roc_curve']['auc'],
            save_path=roc_path
        )
    
    # Получаем и визуализируем важность признаков
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    feature_imp = predictor.get_feature_importances(feature_names)
    
    if feature_imp is not None:
        imp_path = os.path.join(evaluator.results_dir, 'feature_importance.png')
        evaluator.plot_feature_importance(feature_imp, save_path=imp_path) 