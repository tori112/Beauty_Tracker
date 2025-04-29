import pandas as pd
import joblib
import json
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

MODEL_DIR = "models"  # Папка models в корне репозитория
TEMPLATES_PATH = os.path.join(MODEL_DIR, "valid_templates.json")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "best_regressor_tuned_pipeline.pkl")

# Словарь для вычисления method_complexity
method_complexity_map = {
    "Уходовая косметика": 1,
    "Пилинги": 2,
    "Массаж": 2,
    "Тейпирование": 2,
    "Аппаратная косметология": 3,
    "Инъекционная косметология": 5
}

def list_to_text(x):
    """Преобразует список симптомов в строку, разделённую пробелами"""
    if isinstance(x, list):
        return ' '.join(x)
    return str(x)  # На случай, если передали уже строку

def load_models_and_templates():
    """Загружает регрессорный пайплайн и шаблоны с проверкой"""
    required_files = {
        "regressor_pipeline": "best_regressor_tuned_pipeline.pkl",
        "templates": "valid_templates.json"
    }
    
    missing = [name for name, fname in required_files.items() 
               if not os.path.exists(os.path.join(MODEL_DIR, fname))]
    if missing:
        raise FileNotFoundError(f"Отсутствуют файлы: {missing}")

    # Загружаем полный пайплайн регрессора
    regressor_pipeline = joblib.load(os.path.join(MODEL_DIR, required_files["regressor_pipeline"]))
    
    with open(os.path.join(MODEL_DIR, required_files["templates"]), 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    return regressor_pipeline, templates

def get_top_recommendations(problem, skin_type, age_range, symptoms,
                          user_allergies=None, user_contraindications=None, 
                          is_pregnant=False, top_n=3):
    try:
        regressor_pipeline, templates = load_models_and_templates()
        
        # Строгая фильтрация шаблонов по проблеме, типу кожи и возрастному диапазону
        problem_templates = [
            t for t in templates 
            if (t['problem'].replace("ё", "е").lower() == problem.replace("ё", "е").lower() and
                t['skin_type'] == skin_type and
                t['age_range'] == age_range)  # Только точное совпадение
        ]
        if not problem_templates:
            return {"error": f"Нет шаблонов для проблемы '{problem}' с типом кожи '{skin_type}' и возрастным диапазоном '{age_range}'"}

        results = []
        seen_methods = set()  # Для выбора разных методов
        for template in problem_templates:
            method = template['method']
            treatment_type = template['type']
            
            # Проверяем наличие обязательных полей в шаблоне
            required_fields = ['method', 'type']
            missing_fields = [field for field in required_fields if field not in template]
            if missing_fields:
                return {"error": f"В шаблоне для проблемы '{problem}' отсутствуют обязательные поля: {missing_fields}"}

            # Вычисляем method_complexity на основе method
            method_complexity = method_complexity_map.get(method, 1)  # По умолчанию 1, если метод не найден

            # Подготовка входных данных для регрессора
            symptoms_str = list_to_text(symptoms)  # Преобразуем симптомы в строку
            logger.info(f"symptoms_str: {symptoms_str}")
            input_df_reg = pd.DataFrame({
                'problem': [problem],
                'skin_type': [skin_type],
                'age_range': [age_range],
                'symptoms_str': [symptoms_str],  # Используем правильное имя столбца
                'method': [method],
                'type': [treatment_type],
                'method_complexity': [method_complexity]
            })
            logger.info(f"input_df_reg columns: {input_df_reg.columns.tolist()}")
            logger.info(f"input_df_reg data: {input_df_reg.to_dict()}")

            # Предсказываем базовую вероятность с помощью пайплайна
            base_prob = min(regressor_pipeline.predict(input_df_reg)[0], 0.95)  # Макс 95%

            # Применяем корректировки
            def apply_corrections(prob):
                # Аллергии на компоненты
                if user_allergies and any(a.replace("Аллергия на ", "").lower() in [ing.lower() for ing in template.get('active_ingredients', [])] for a in user_allergies):
                    prob *= 0.6
                    
                # Беременность
                if is_pregnant and template.get('contraindicated_during_pregnancy', False):
                    prob *= 0.7
                    
                # Противопоказания
                if user_contraindications and any(c.lower() in template.get('contraindications', '').lower() for c in user_contraindications):
                    prob *= 0.8
                    
                return max(prob, 0.1)  # Минимум 10%

            final_prob = round(apply_corrections(base_prob) * 100, 0)  # Конвертируем в проценты и округляем до целого

            # Формируем результат
            result = {
                'method': method,
                'type': treatment_type,
                'success_prob': final_prob,
                'template': template.get('template', 'Описание отсутствует'),
                'expected_effect': ', '.join(template.get('effects', ['Не указан'])),
                'course_duration': str(template.get('course_duration', 'Не указана')),
                'active_ingredients': template.get('active_ingredients', []),
                'contraindications': template.get('contraindications', 'Нет').split('\n'),
                'base_prob': round(base_prob, 2)
            }
            
            if final_prob < 30:
                result['warning'] = "Низкая эффективность из-за противопоказаний"
                
            results.append(result)

        # Сортируем по вероятности и выбираем топ-N с разными методами
        top_recommendations = []
        sorted_results = sorted(results, key=lambda x: -x['success_prob'])
        for result in sorted_results:
            if result['method'] not in seen_methods:
                seen_methods.add(result['method'])
                top_recommendations.append(result)
                if len(top_recommendations) >= top_n:
                    break
        
        if not top_recommendations:
            return {"error": f"Не удалось найти рекомендации для проблемы: {problem}"}
        
        return {
            'problem': problem,
            'recommendations': top_recommendations
        }

    except Exception as e:
        logger.error(f"Ошибка в get_top_recommendations: {str(e)}")
        return {"error": f"Ошибка при формировании рекомендаций: {str(e)}"}
    
def predict_for_multiple_problems(problems, skin_type, age_range, symptoms,
                                user_allergies=None, user_contraindications=None,
                                is_pregnant=False, top_per_problem=3):
    """Обрабатывает несколько проблем и возвращает рекомендации для каждой"""
    if isinstance(problems, str):
        problems = [problems]
        
    all_results = []
    for problem in problems:
        result = get_top_recommendations(
            problem=problem,
            skin_type=skin_type,
            age_range=age_range,
            symptoms=symptoms,
            user_allergies=user_allergies,
            user_contraindications=user_contraindications,
            is_pregnant=is_pregnant,
            top_n=top_per_problem
        )
        all_results.append(result)
    
    return all_results