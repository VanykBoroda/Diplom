import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Класс для генерации отчетов об отчислении студентов.
    """
    
    def __init__(self, reports_dir: str = "reports"):
        """
        Инициализация генератора отчетов.
        
        Args:
            reports_dir: Директория для хранения отчетов
        """
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
    
    def generate_risk_report(self, student_risks: pd.DataFrame, 
                           model_info: Dict[str, Any],
                           report_title: str = "Отчет о рисках отчисления",
                           save_path: Optional[str] = None) -> str:
        """
        Генерация отчета о рисках отчисления в формате HTML.
        
        Args:
            student_risks: DataFrame с данными о рисках студентов
            model_info: Информация о модели и метриках
            report_title: Заголовок отчета
            save_path: Путь для сохранения отчета
            
        Returns:
            str: Путь к сгенерированному отчету
        """
        # Если путь не указан, генерируем имя файла
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.reports_dir, f"risk_report_{timestamp}.html")
        
        # Создаем HTML-отчет
        html_parts = []
        
        # Заголовок и метаданные
        html_parts.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4B0082; color: white; padding: 10px 20px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .risk-high {{ color: red; font-weight: bold; }}
                .risk-medium {{ color: orange; }}
                .risk-low {{ color: green; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #666; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_title}</h1>
                <p>Дата: {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
            </div>
        """)
        
        # Сводная информация
        total_students = len(student_risks)
        risk_levels = student_risks['risk_level'].value_counts()
        high_risk_count = risk_levels.get('Высокий', 0)
        medium_risk_count = risk_levels.get('Средний', 0)
        low_risk_count = risk_levels.get('Низкий', 0)
        
        html_parts.append(f"""
            <div class="section">
                <h2>Сводная информация</h2>
                <p><strong>Всего студентов:</strong> {total_students}</p>
                <p><strong>Студентов с высоким риском отчисления:</strong> {high_risk_count} ({high_risk_count/total_students*100:.1f}%)</p>
                <p><strong>Студентов со средним риском отчисления:</strong> {medium_risk_count} ({medium_risk_count/total_students*100:.1f}%)</p>
                <p><strong>Студентов с низким риском отчисления:</strong> {low_risk_count} ({low_risk_count/total_students*100:.1f}%)</p>
                
                <h3>Информация о модели</h3>
                <p><strong>Модель:</strong> {model_info.get('model_name', 'Неизвестно')}</p>
                <p><strong>Точность (Accuracy):</strong> {model_info.get('accuracy', 0):.4f}</p>
                <p><strong>F1-мера:</strong> {model_info.get('f1', 0):.4f}</p>
            </div>
        """)
        
        # Визуализация распределения рисков
        risk_level_order = ['Высокий', 'Средний', 'Низкий']
        
        # Создаем график
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='risk_level', data=student_risks, order=risk_level_order)
        ax.set_xlabel('Уровень риска')
        ax.set_ylabel('Количество студентов')
        ax.set_title('Распределение уровней риска отчисления')
        
        # Добавляем метки значений на столбцы
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', xytext=(0, 10), 
                       textcoords='offset points')
        
        # Сохраняем график
        chart_path = os.path.join(self.reports_dir, "risk_distribution.png")
        plt.savefig(chart_path)
        plt.close()
        
        # Добавляем график в отчет
        html_parts.append(f"""
            <div class="chart">
                <h2>Распределение рисков отчисления</h2>
                <img src="{os.path.basename(chart_path)}" alt="Распределение рисков" style="max-width: 100%;">
            </div>
        """)
        
        # Гистограмма вероятностей
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(student_risks['dropout_probability'], bins=20, kde=True)
        ax.axvline(x=0.3, color='green', linestyle='--', label='Низкий риск')
        ax.axvline(x=0.6, color='orange', linestyle='--', label='Средний риск')
        ax.set_xlabel('Вероятность отчисления')
        ax.set_ylabel('Количество студентов')
        ax.set_title('Распределение вероятностей отчисления')
        ax.legend()
        
        # Сохраняем график
        hist_path = os.path.join(self.reports_dir, "probability_distribution.png")
        plt.savefig(hist_path)
        plt.close()
        
        # Добавляем график в отчет
        html_parts.append(f"""
            <div class="chart">
                <h2>Распределение вероятностей отчисления</h2>
                <img src="{os.path.basename(hist_path)}" alt="Распределение вероятностей" style="max-width: 100%;">
            </div>
        """)
        
        # Список студентов с высоким риском
        high_risk_students = student_risks[student_risks['risk_level'] == 'Высокий'].sort_values(
            by='dropout_probability', ascending=False)
        
        if not high_risk_students.empty:
            html_parts.append("""
                <div class="section">
                    <h2>Студенты с высоким риском отчисления</h2>
                    <table>
                        <tr>
                            <th>ID студента</th>
                            <th>Вероятность отчисления</th>
                            <th>Уровень риска</th>
                        </tr>
            """)
            
            for _, row in high_risk_students.iterrows():
                html_parts.append(f"""
                    <tr>
                        <td>{row['student_id']}</td>
                        <td>{row['dropout_probability']:.4f}</td>
                        <td class="risk-high">{row['risk_level']}</td>
                    </tr>
                """)
            
            html_parts.append("</table></div>")
        
        # Список студентов со средним риском
        medium_risk_students = student_risks[student_risks['risk_level'] == 'Средний'].sort_values(
            by='dropout_probability', ascending=False)
        
        if not medium_risk_students.empty:
            html_parts.append("""
                <div class="section">
                    <h2>Студенты со средним риском отчисления</h2>
                    <table>
                        <tr>
                            <th>ID студента</th>
                            <th>Вероятность отчисления</th>
                            <th>Уровень риска</th>
                        </tr>
            """)
            
            for _, row in medium_risk_students.iterrows():
                html_parts.append(f"""
                    <tr>
                        <td>{row['student_id']}</td>
                        <td>{row['dropout_probability']:.4f}</td>
                        <td class="risk-medium">{row['risk_level']}</td>
                    </tr>
                """)
            
            html_parts.append("</table></div>")
        
        # Рекомендации
        html_parts.append("""
            <div class="section">
                <h2>Рекомендации</h2>
                <ol>
                    <li>Организовать индивидуальные встречи со студентами из группы высокого риска для выявления проблем и предоставления необходимой поддержки.</li>
                    <li>Рассмотреть возможность дополнительных консультаций по сложным предметам для студентов с высоким и средним риском отчисления.</li>
                    <li>Провести анализ учебной нагрузки и, при необходимости, скорректировать учебный план.</li>
                    <li>Организовать психологическую поддержку для студентов с проблемами адаптации.</li>
                    <li>Регулярно отслеживать прогресс студентов из групп риска.</li>
                </ol>
            </div>
        """)
        
        # Завершаем документ
        html_parts.append("""
            <div class="footer">
                <p>Отчет сгенерирован системой прогнозирования отчисления студентов Московского университета имени С.Ю. Витте</p>
            </div>
        </body>
        </html>
        """)
        
        # Сохраняем HTML-отчет
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
            
        # Копируем изображения в ту же директорию, что и отчет
        import shutil
        target_dir = os.path.dirname(save_path)
        if target_dir != self.reports_dir:
            shutil.copy(chart_path, target_dir)
            shutil.copy(hist_path, target_dir)
            
        logger.info(f"Отчет о рисках отчисления сгенерирован: {save_path}")
        return save_path
    
    def generate_comparative_report(self, risk_data1: pd.DataFrame, risk_data2: pd.DataFrame,
                                  date1: str, date2: str,
                                  save_path: Optional[str] = None) -> str:
        """
        Генерация отчета, сравнивающего риски отчисления в две разные даты.
        
        Args:
            risk_data1: DataFrame с данными о рисках за первую дату
            risk_data2: DataFrame с данными о рисках за вторую дату
            date1: Первая дата в формате строки
            date2: Вторая дата в формате строки
            save_path: Путь для сохранения отчета
            
        Returns:
            str: Путь к сгенерированному отчету
        """
        # Если путь не указан, генерируем имя файла
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.reports_dir, f"comparative_report_{timestamp}.html")
        
        # Объединяем данные
        merged_data = pd.merge(
            risk_data1[['student_id', 'dropout_probability', 'risk_level']],
            risk_data2[['student_id', 'dropout_probability', 'risk_level']],
            on='student_id',
            suffixes=('_before', '_after')
        )
        
        # Рассчитываем изменения
        merged_data['change'] = merged_data['dropout_probability_after'] - merged_data['dropout_probability_before']
        
        # Создаем HTML-отчет
        html_parts = []
        
        # Заголовок и метаданные
        html_parts.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Сравнительный отчет по рискам отчисления</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4B0082; color: white; padding: 10px 20px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .improved {{ color: green; }}
                .worsened {{ color: red; }}
                .unchanged {{ color: gray; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #666; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Сравнительный отчет по рискам отчисления</h1>
                <p>Сравнение данных за периоды: {date1} и {date2}</p>
                <p>Дата формирования: {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
            </div>
        """)
        
        # Сводная информация об изменениях
        total_students = len(merged_data)
        improved_count = len(merged_data[merged_data['change'] < -0.05])
        worsened_count = len(merged_data[merged_data['change'] > 0.05])
        unchanged_count = total_students - improved_count - worsened_count
        
        mean_before = merged_data['dropout_probability_before'].mean()
        mean_after = merged_data['dropout_probability_after'].mean()
        mean_change = mean_after - mean_before
        
        html_parts.append(f"""
            <div class="section">
                <h2>Сводная информация об изменениях</h2>
                <p><strong>Всего студентов:</strong> {total_students}</p>
                <p><strong>Студентов с улучшением показателей:</strong> {improved_count} ({improved_count/total_students*100:.1f}%)</p>
                <p><strong>Студентов с ухудшением показателей:</strong> {worsened_count} ({worsened_count/total_students*100:.1f}%)</p>
                <p><strong>Студентов без существенных изменений:</strong> {unchanged_count} ({unchanged_count/total_students*100:.1f}%)</p>
                <p><strong>Средний риск до:</strong> {mean_before:.4f}</p>
                <p><strong>Средний риск после:</strong> {mean_after:.4f}</p>
                <p><strong>Изменение среднего риска:</strong> <span class="{'improved' if mean_change < 0 else 'worsened' if mean_change > 0 else 'unchanged'}">{mean_change:.4f}</span></p>
            </div>
        """)
        
        # Визуализация изменений
        plt.figure(figsize=(10, 6))
        categories = ['Улучшение', 'Без изменений', 'Ухудшение']
        values = [improved_count, unchanged_count, worsened_count]
        colors = ['green', 'gray', 'red']
        
        ax = plt.bar(categories, values, color=colors)
        plt.xlabel('Категория изменения')
        plt.ylabel('Количество студентов')
        plt.title('Распределение изменений рисков отчисления')
        
        # Сохраняем график
        chart_path = os.path.join(self.reports_dir, "changes_distribution.png")
        plt.savefig(chart_path)
        plt.close()
        
        # Добавляем график в отчет
        html_parts.append(f"""
            <div class="chart">
                <h2>Распределение изменений рисков отчисления</h2>
                <img src="{os.path.basename(chart_path)}" alt="Распределение изменений" style="max-width: 100%;">
            </div>
        """)
        
        # Топ студентов с наибольшим улучшением
        improved_students = merged_data.sort_values(by='change')
        top_improved = improved_students.head(10)
        
        html_parts.append("""
            <div class="section">
                <h2>Топ-10 студентов с наибольшим улучшением</h2>
                <table>
                    <tr>
                        <th>ID студента</th>
                        <th>Риск до</th>
                        <th>Риск после</th>
                        <th>Изменение</th>
                    </tr>
        """)
        
        for _, row in top_improved.iterrows():
            html_parts.append(f"""
                <tr>
                    <td>{row['student_id']}</td>
                    <td>{row['dropout_probability_before']:.4f}</td>
                    <td>{row['dropout_probability_after']:.4f}</td>
                    <td class="improved">{row['change']:.4f}</td>
                </tr>
            """)
        
        html_parts.append("</table></div>")
        
        # Топ студентов с наибольшим ухудшением
        top_worsened = improved_students.tail(10).iloc[::-1]
        
        html_parts.append("""
            <div class="section">
                <h2>Топ-10 студентов с наибольшим ухудшением</h2>
                <table>
                    <tr>
                        <th>ID студента</th>
                        <th>Риск до</th>
                        <th>Риск после</th>
                        <th>Изменение</th>
                    </tr>
        """)
        
        for _, row in top_worsened.iterrows():
            html_parts.append(f"""
                <tr>
                    <td>{row['student_id']}</td>
                    <td>{row['dropout_probability_before']:.4f}</td>
                    <td>{row['dropout_probability_after']:.4f}</td>
                    <td class="worsened">{row['change']:.4f}</td>
                </tr>
            """)
        
        html_parts.append("</table></div>")
        
        # Рекомендации
        html_parts.append("""
            <div class="section">
                <h2>Рекомендации</h2>
                <ol>
                    <li>Провести детальный анализ факторов, способствовавших улучшению показателей у студентов с наибольшим прогрессом.</li>
                    <li>Разработать индивидуальные планы поддержки для студентов с наибольшим ухудшением показателей.</li>
                    <li>Масштабировать успешные практики на всю студенческую группу.</li>
                    <li>Пересмотреть программы поддержки, если общий тренд показывает ухудшение рисков отчисления.</li>
                    <li>Регулярно проводить сравнительный анализ для отслеживания эффективности принимаемых мер.</li>
                </ol>
            </div>
        """)
        
        # Завершаем документ
        html_parts.append("""
            <div class="footer">
                <p>Отчет сгенерирован системой прогнозирования отчисления студентов Московского университета имени С.Ю. Витте</p>
            </div>
        </body>
        </html>
        """)
        
        # Сохраняем HTML-отчет
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
            
        # Копируем изображения в ту же директорию, что и отчет
        import shutil
        target_dir = os.path.dirname(save_path)
        if target_dir != self.reports_dir:
            shutil.copy(chart_path, target_dir)
            
        logger.info(f"Сравнительный отчет сгенерирован: {save_path}")
        return save_path


if __name__ == "__main__":
    # Пример использования
    import numpy as np
    
    # Создаем тестовые данные
    n_samples = 100
    
    # Данные о рисках
    student_ids = [f'S{i:04d}' for i in range(n_samples)]
    dropout_probs = np.random.uniform(0, 1, n_samples)
    
    # Определяем уровни риска
    risk_levels = pd.cut(
        dropout_probs, 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Низкий', 'Средний', 'Высокий']
    )
    
    # Создаем DataFrame
    student_risks = pd.DataFrame({
        'student_id': student_ids,
        'dropout_probability': dropout_probs,
        'prediction': (dropout_probs > 0.5).astype(int),
        'risk_level': risk_levels
    })
    
    # Информация о модели
    model_info = {
        'model_name': 'Random Forest Classifier',
        'accuracy': 0.87,
        'precision': 0.84,
        'recall': 0.81,
        'f1': 0.82
    }
    
    # Генерируем отчет
    generator = ReportGenerator()
    report_path = generator.generate_risk_report(student_risks, model_info)
    
    print(f"Отчет сгенерирован: {report_path}")
    
    # Создаем данные для сравнительного отчета
    dropout_probs2 = np.clip(dropout_probs + np.random.normal(0, 0.2, n_samples), 0, 1)
    
    risk_levels2 = pd.cut(
        dropout_probs2, 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Низкий', 'Средний', 'Высокий']
    )
    
    student_risks2 = pd.DataFrame({
        'student_id': student_ids,
        'dropout_probability': dropout_probs2,
        'prediction': (dropout_probs2 > 0.5).astype(int),
        'risk_level': risk_levels2
    })
    
    # Генерируем сравнительный отчет
    comp_report_path = generator.generate_comparative_report(
        student_risks, student_risks2,
        '2025-01-01', '2025-02-01'
    )
    
    print(f"Сравнительный отчет сгенерирован: {comp_report_path}") 