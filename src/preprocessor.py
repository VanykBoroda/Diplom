import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from typing import Tuple, List, Dict, Any, Optional, Union

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StudentDataPreprocessor:
    """
    Класс для предобработки данных о студентах перед использованием в моделях.
    Выполняет:
    - Очистку данных
    - Обработку пропущенных значений
    - Кодирование категориальных признаков
    - Масштабирование числовых признаков
    - Создание новых признаков
    """
    
    def __init__(self):
        """Инициализация предобработчика данных."""
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        
    def fit(self, data: pd.DataFrame, target_column: str = 'dropout') -> 'StudentDataPreprocessor':
        """
        Обучение предобработчика на данных.
        
        Args:
            data: DataFrame с исходными данными
            target_column: Название целевой переменной
            
        Returns:
            self: Обученный предобработчик
        """
        logger.info("Обучение предобработчика данных")
        
        # Копируем данные, чтобы не менять оригинал
        data = data.copy()
        
        # Разделяем признаки на числовые и категориальные
        categorical_cols, numeric_cols = self._identify_column_types(data)
        
        # Удаляем целевую переменную из списков признаков, если она там есть
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
            
        # Создаем pipeline для предобработки данных
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.pipeline = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
        # Обучаем pipeline
        self.pipeline.fit(data.drop(columns=[target_column], errors='ignore'))
        
        # Если целевая переменная существует и она категориальная, обучаем LabelEncoder
        if target_column in data.columns and data[target_column].dtype == 'object':
            self.label_encoder.fit(data[target_column])
            
        logger.info("Предобработчик обучен успешно")
        return self
    
    def transform(self, data: pd.DataFrame, target_column: Optional[str] = 'dropout') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Преобразование данных с использованием обученного предобработчика.
        
        Args:
            data: DataFrame с исходными данными
            target_column: Название целевой переменной (может быть None)
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Кортеж (X, y), где X - преобразованные признаки,
                                                   y - преобразованная целевая переменная (может быть None)
        """
        logger.info("Применение предобработки к данным")
        
        # Копируем данные, чтобы не менять оригинал
        data = data.copy()
        
        # Проверяем, что pipeline был обучен
        if self.pipeline is None:
            raise ValueError("Необходимо сначала обучить предобработчик с помощью метода fit()")
            
        # Преобразуем признаки
        X = self.pipeline.transform(data.drop(columns=[target_column], errors='ignore'))
        
        # Преобразуем целевую переменную, если она есть
        y = None
        if target_column in data.columns:
            if data[target_column].dtype == 'object':
                y = self.label_encoder.transform(data[target_column])
            else:
                y = data[target_column].values
                
        logger.info(f"Данные преобразованы. Размерность X: {X.shape}")
        return X, y
    
    def fit_transform(self, data: pd.DataFrame, target_column: str = 'dropout') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Обучение предобработчика и преобразование данных за один шаг.
        
        Args:
            data: DataFrame с исходными данными
            target_column: Название целевой переменной
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Кортеж (X, y) с преобразованными данными
        """
        self.fit(data, target_column)
        return self.transform(data, target_column)
    
    def _identify_column_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Определение типов столбцов (категориальные и числовые)
        
        Args:
            data: DataFrame с исходными данными
            
        Returns:
            Tuple[List[str], List[str]]: Кортеж (categorical_cols, numeric_cols) с названиями столбцов
        """
        categorical_cols = []
        numeric_cols = []
        
        # Определяем тип каждого столбца
        for col in data.columns:
            # Пропускаем столбцы с идентификаторами
            if col.lower() in ['id', 'student_id', 'identifier']:
                continue
                
            # Определяем тип столбца
            if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(data[col]):
                numeric_cols.append(col)
                
        logger.debug(f"Определены категориальные признаки: {categorical_cols}")
        logger.debug(f"Определены числовые признаки: {numeric_cols}")
        
        return categorical_cols, numeric_cols
    
    def get_feature_names(self) -> List[str]:
        """
        Получение названий признаков после преобразования
        
        Returns:
            List[str]: Список названий признаков
        """
        if hasattr(self.pipeline, 'get_feature_names_out'):
            return self.pipeline.get_feature_names_out()
        
        features = []
        for name, trans, cols in self.pipeline.transformers_:
            if name == 'num':
                features.extend(cols)
            elif name == 'cat':
                for col in cols:
                    features.extend([f"{col}_{cat}" for cat in trans.named_steps['onehot'].categories_[cols.index(col)]])
                    
        return features
    
    def create_additional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создание дополнительных признаков на основе существующих
        
        Args:
            data: DataFrame с исходными данными
            
        Returns:
            pd.DataFrame: DataFrame с добавленными признаками
        """
        logger.info("Создание дополнительных признаков")
        
        # Копируем данные, чтобы не менять оригинал
        df = data.copy()
        
        # Пример создания новых признаков
        # Признаки ниже должны быть адаптированы под реальные данные
        
        # 1. Соотношение пропусков занятий к общему числу занятий
        if 'absences' in df.columns and 'total_classes' in df.columns:
            df['absence_rate'] = df['absences'] / df['total_classes']
            
        # 2. Флаг низкой успеваемости
        if 'gpa' in df.columns:
            df['low_performance'] = (df['gpa'] < 3.0).astype(int)
            
        # 3. Возрастная группа
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 18, 21, 25, 30, 100], 
                labels=['under_18', '18-21', '22-25', '26-30', 'over_30']
            )
            
        # 4. Комбинированные признаки на основе факультета и года обучения
        if 'faculty' in df.columns and 'year_of_study' in df.columns:
            df['faculty_year'] = df['faculty'] + '_' + df['year_of_study'].astype(str)
            
        # Другие признаки могут быть добавлены по мере необходимости
        
        logger.info(f"Создано {len(df.columns) - len(data.columns)} новых признаков")
        return df


if __name__ == "__main__":
    # Пример использования
    from data_loader import DataLoader
    import os
    
    loader = DataLoader()
    sample_data = loader.generate_sample_data(1000)
    
    preprocessor = StudentDataPreprocessor()
    X, y = preprocessor.fit_transform(sample_data)
    
    print(f"Размерность признаков после преобразования: {X.shape}")
    if y is not None:
        print(f"Размерность целевой переменной: {y.shape}")
        print(f"Распределение классов: {np.bincount(y)}") 