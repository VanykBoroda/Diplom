import os
import pandas as pd
import kaggle
from typing import Union, Optional, Dict, Any
import json
import logging
import time
import requests

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Класс для загрузки данных из различных источников:
    - Локальные файлы (CSV, Excel, JSON)
    - Kaggle датасеты
    - Другие облачные хранилища
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Инициализация загрузчика данных
        
        Args:
            data_dir: Директория для хранения данных
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_local_file(self, file_path: str) -> pd.DataFrame:
        """
        Загрузка данных из локального файла
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            DataFrame с загруженными данными
        """
        logger.info(f"Загрузка данных из файла: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                # Пробуем разные разделители и кодировки
                for encoding in ['utf-8', 'cp1251', 'latin1']:
                    for sep in [',', ';', '\t']:
                        try:
                            return pd.read_csv(file_path, sep=sep, encoding=encoding)
                        except:
                            continue
                raise ValueError(f"Не удалось загрузить CSV файл: {file_path}")
                
            elif file_extension in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
                
            elif file_extension == '.json':
                return pd.read_json(file_path)
                
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Сохранение данных в файл
        
        Args:
            data: DataFrame с данными
            file_path: Путь для сохранения файла
        """
        logger.info(f"Сохранение данных в файл: {file_path}")
        
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                data.to_csv(file_path, index=False, encoding='utf-8')
                
            elif file_extension in ['.xls', '.xlsx']:
                data.to_excel(file_path, index=False)
                
            elif file_extension == '.json':
                data.to_json(file_path, orient='records')
                
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
                
            logger.info(f"Данные успешно сохранены в {file_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных в {file_path}: {str(e)}")
            raise
    
    def download_from_kaggle(self, dataset: str, path: Optional[str] = None) -> str:
        """
        Загрузка датасета с Kaggle
        
        Args:
            dataset: Имя датасета в формате 'username/dataset-name'
            path: Путь для сохранения, по умолчанию используется self.data_dir
            
        Returns:
            Путь к загруженному датасету
        """
        if path is None:
            path = self.data_dir
            
        logger.info(f"Загрузка датасета с Kaggle: {dataset}")
        
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
            logger.info(f"Датасет {dataset} успешно загружен в {path}")
            return path
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета с Kaggle: {str(e)}")
            raise
    
    def load_student_data(self, file_path: str) -> pd.DataFrame:
        """
        Специализированная функция для загрузки данных о студентах
        
        Args:
            file_path: Путь к файлу с данными о студентах
            
        Returns:
            DataFrame с данными о студентах
        """
        logger.info(f"Загрузка данных о студентах из файла: {file_path}")
        data = self.load_local_file(file_path)
        
        # Здесь можно добавить проверку корректности данных и базовую предобработку
        # TODO: Добавить валидацию структуры данных о студентах
        
        return data
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Генерация тестовых данных для разработки и отладки
        
        Args:
            n_samples: Количество записей для генерации
            
        Returns:
            DataFrame с сгенерированными данными
        """
        import numpy as np
        from sklearn.datasets import make_classification
        
        logger.info(f"Генерация тестовых данных: {n_samples} записей")
        
        # Создаем синтетические признаки
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Преобразуем в датафрейм и добавляем названия
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Добавляем демографические и образовательные признаки
        df['student_id'] = [f'S{i:05d}' for i in range(n_samples)]
        df['age'] = np.random.randint(17, 30, size=n_samples)
        df['gender'] = np.random.choice(['M', 'F'], size=n_samples)
        df['gpa'] = np.random.uniform(2.0, 5.0, size=n_samples)
        df['credits'] = np.random.randint(0, 240, size=n_samples)
        df['absences'] = np.random.randint(0, 30, size=n_samples)
        df['scholarship'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        df['employment'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        df['year_of_study'] = np.random.randint(1, 5, size=n_samples)
        df['faculty'] = np.random.choice(['Economics', 'Law', 'Computer Science', 
                                         'Management', 'Psychology'], 
                                         size=n_samples)
        
        # Целевая переменная - отчислен или нет
        df['dropout'] = y
        
        logger.info("Тестовые данные успешно сгенерированы")
        return df
    
    def download_from_google_drive(self, file_id: str, output_path: Optional[str] = None) -> str:
        """
        Загрузка файла с Google Drive
        
        Args:
            file_id: ID файла в Google Drive
            output_path: Путь для сохранения файла, по умолчанию используется self.data_dir
            
        Returns:
            Путь к загруженному файлу
        """
        if output_path is None:
            # Генерируем имя файла на основе текущего времени
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.data_dir, f"gdrive_file_{timestamp}")
        
        logger.info(f"Загрузка файла с Google Drive (ID: {file_id})")
        
        try:
            # Использование прямого URL для загрузки
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Загрузка файла
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Определяем расширение файла, если возможно
            content_disposition = response.headers.get('content-disposition')
            filename = None
            
            if content_disposition:
                import re
                filename_match = re.findall('filename="(.+)"', content_disposition)
                if filename_match:
                    filename = filename_match[0]
            
            if filename:
                output_path = os.path.join(os.path.dirname(output_path), filename)
            elif not os.path.splitext(output_path)[1]:
                # Если не удалось определить имя файла и нет расширения, добавляем .bin
                output_path += '.bin'
            
            # Сохраняем файл
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Файл успешно загружен и сохранен в {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла с Google Drive: {str(e)}")
            raise
    
    def download_from_dropbox(self, url: str, output_path: Optional[str] = None) -> str:
        """
        Загрузка файла с Dropbox
        
        Args:
            url: Публичная ссылка на файл в Dropbox
            output_path: Путь для сохранения файла, по умолчанию используется self.data_dir
            
        Returns:
            Путь к загруженному файлу
        """
        if output_path is None:
            # Генерируем имя файла на основе текущего времени
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.data_dir, f"dropbox_file_{timestamp}")
        
        logger.info(f"Загрузка файла с Dropbox (URL: {url})")
        
        try:
            # Заменяем www.dropbox.com на dl.dropboxusercontent.com для прямой загрузки
            download_url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
            
            # Загрузка файла
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Определяем имя файла из URL, если возможно
            filename = os.path.basename(url.split('?')[0])
            
            if filename:
                output_path = os.path.join(os.path.dirname(output_path), filename)
            elif not os.path.splitext(output_path)[1]:
                # Если не удалось определить имя файла и нет расширения, добавляем .bin
                output_path += '.bin'
            
            # Сохраняем файл
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Файл успешно загружен и сохранен в {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла с Dropbox: {str(e)}")
            raise
    
    def download_from_onedrive(self, url: str, output_path: Optional[str] = None) -> str:
        """
        Загрузка файла с OneDrive
        
        Args:
            url: Публичная ссылка на файл в OneDrive
            output_path: Путь для сохранения файла, по умолчанию используется self.data_dir
            
        Returns:
            Путь к загруженному файлу
        """
        if output_path is None:
            # Генерируем имя файла на основе текущего времени
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.data_dir, f"onedrive_file_{timestamp}")
        
        logger.info(f"Загрузка файла с OneDrive (URL: {url})")
        
        try:
            # Преобразуем URL для прямой загрузки
            if '1drv.ms' in url or 'onedrive.live.com' in url:
                # Переходим по короткой ссылке и получаем фактический URL
                session = requests.Session()
                response = session.get(url, allow_redirects=True)
                url = response.url
                
            # Заменяем "view.aspx" на "download.aspx" для прямой загрузки
            download_url = url.replace('view.aspx', 'download.aspx')
            
            # Загрузка файла
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Определяем имя файла из заголовков, если возможно
            content_disposition = response.headers.get('content-disposition')
            filename = None
            
            if content_disposition:
                import re
                filename_match = re.findall('filename="(.+)"', content_disposition)
                if filename_match:
                    filename = filename_match[0]
            
            if filename:
                output_path = os.path.join(os.path.dirname(output_path), filename)
            elif not os.path.splitext(output_path)[1]:
                # Если не удалось определить имя файла и нет расширения, добавляем .bin
                output_path += '.bin'
            
            # Сохраняем файл
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Файл успешно загружен и сохранен в {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла с OneDrive: {str(e)}")
            raise
    
    def download_from_aws_s3(self, bucket_name: str, object_key: str, 
                           aws_access_key_id: str, aws_secret_access_key: str,
                           region_name: str = 'us-east-1',
                           output_path: Optional[str] = None) -> str:
        """
        Загрузка файла с Amazon S3
        
        Args:
            bucket_name: Имя бакета S3
            object_key: Ключ объекта (путь к файлу в бакете)
            aws_access_key_id: Идентификатор ключа доступа AWS
            aws_secret_access_key: Секретный ключ доступа AWS
            region_name: Регион AWS
            output_path: Путь для сохранения файла, по умолчанию используется self.data_dir
            
        Returns:
            Путь к загруженному файлу
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            if output_path is None:
                # Генерируем имя файла из ключа объекта
                filename = os.path.basename(object_key)
                if not filename:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"s3_file_{timestamp}"
                
                output_path = os.path.join(self.data_dir, filename)
            
            logger.info(f"Загрузка файла с Amazon S3 (Bucket: {bucket_name}, Key: {object_key})")
            
            # Создаем клиент S3
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
            
            # Загружаем файл
            s3_client.download_file(bucket_name, object_key, output_path)
            
            logger.info(f"Файл успешно загружен и сохранен в {output_path}")
            return output_path
            
        except ImportError:
            logger.error("Для загрузки данных с AWS S3 требуется установить boto3. Используйте 'pip install boto3'")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла с AWS S3: {str(e)}")
            raise


if __name__ == "__main__":
    # Пример использования
    loader = DataLoader()
    
    # Генерация и сохранение тестовых данных
    sample_data = loader.generate_sample_data(1000)
    sample_path = os.path.join(loader.data_dir, "sample_students.csv")
    loader.save_data(sample_data, sample_path)
    
    # Загрузка сохраненных данных
    loaded_data = loader.load_local_file(sample_path)
    print(f"Загружено {len(loaded_data)} записей")
    print(loaded_data.head()) 