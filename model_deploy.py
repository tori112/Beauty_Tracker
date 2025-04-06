import os
import gdown
import joblib
import numpy as np
import pandas as pd
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import list_to_text, custom_tokenizer

class SkincareModel:
    def __init__(self, 
                 model_path="models/ruT5",
                 pkl_path="models/best_models.pkl",
                 similarity_threshold=0.45):
        self.model_path = model_path
        self.pkl_path = pkl_path
        self.similarity_threshold = similarity_threshold
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.regressor = None
        self.vectorizer = None
        self.label_encoders = None
        self.is_trained = False
        self.used_combinations = set()

        self._initialize_attributes()

    def _download_models(self):
        try:
            required_files = [
                "spiece.model", "training_args.bin", "tokenizer_config.json",
                "generation_config.json", "spiece.model.old", "config.json",
                "added_tokens.json", "special_tokens_map.json", "model.safetensors"
            ]
            t5_files_present = all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files)
            print(f"Папка {self.model_path} существует: {os.path.exists(self.model_path)}")
            print(f"Все необходимые файлы T5 присутствуют: {t5_files_present}")
            if not t5_files_present:
                print("Скачиваем T5 модель...")
                gdown.download_folder(
                    "https://drive.google.com/drive/folders/1_zIAEkqBeR8_yS6AyfIzjLNZpNtWULfW",
                    output=self.model_path,
                    quiet=False
                )
                print("T5 модель успешно скачана.")
                print("Содержимое папки models/ruT5 после скачивания:")
                for root, dirs, files in os.walk(self.model_path):
                    print(f"Путь: {root}, Файлы: {files}")
            
            if not os.path.exists(self.pkl_path):
                print("Скачиваем pkl файл...")
                gdown.download(
                    "https://drive.google.com/uc?id=1SdvXJDBxrUS_d_Y0p5qlyU1Cw1ONc5wg",
                    self.pkl_path,
                    quiet=False
                )
                print("pkl файл успешно скачан.")
        except Exception as e:
            print(f"Ошибка при скачивании моделей: {str(e)}")

    def _initialize_attributes(self):
        self.problem_to_recommendation = {
            'Обезвоженность': [
                {'method': 'Уходовая косметика', 'type': 'Крем', 'effectiveness': 'Высокая', 'course_duration': 30},
                {'method': 'Уходовая косметика', 'type': 'Сыворотка', 'effectiveness': 'Высокая', 'course_duration': 30}
            ],
            'Чёрные точки': [
                {'method': 'Аппаратная косметология', 'type': 'Ультразвуковая чистка', 'effectiveness': 'Высокая', 'course_duration': 5},
                {'method': 'Уходовая косметика', 'type': 'Пилинг', 'effectiveness': 'Средняя', 'course_duration': 14}
            ],
            'Морщины': [
                {'method': 'Аппаратная косметология', 'type': 'RF-лифтинг', 'effectiveness': 'Высокая', 'course_duration': 10},
                {'method': 'Инъекционная косметология', 'type': 'Ботулинотерапия', 'effectiveness': 'Высокая', 'course_duration': 1}
            ],
            'Жирный блеск': [
                {'method': 'Уходовая косметика', 'type': 'Тоник', 'effectiveness': 'Средняя', 'course_duration': 30},
                {'method': 'Пилинги', 'type': 'Поверхностный', 'effectiveness': 'Высокая', 'course_duration': 7}
            ]
        }
        
        self.validation_rules = {
            'keywords': {
                'Ультразвуковая чистка': ['ультразвук', 'чистка', 'поры'],
                'Крем': ['наносить', 'крем', 'утро', 'салициловая кислота'],
                'Сыворотка': ['наносить', 'сыворотка', 'вечер'],
                'Пилинг': ['использовать', 'пилинг', 'неделя'],
                'RF-лифтинг': ['курс', 'процедур', 'косметолог'],
                'Ботулинотерапия': ['инъекция', 'косметолог', 'морщины'],
                'Миофасциальный': ['массаж', 'сеанс', 'профессионал']
            },
            'forbidden_pairs': [
                (['инъекц', 'укол'], 'Уходовая косметика')
            ]
        }

        self.treatment_categories = {
            'Уходовая косметика': ["Крем", "Тоник", "Сыворотка", "Пилинг", "Маска", "Скраб", 'Гель', 'Патчи', 'Мыло'],
            'Аппаратная косметология': ["RF-лифтинг", "Микротоки", "Ультразвуковая чистка", 
                        "Фотоомоложение", "Лазерная терапия"],
            'Инъекционная косметология': ["Мезотерапия", "Биоревитализация", "Ботулинотерапия", 
                        "Контурная пластика"],
            'Пилинги': ['Поверхностный', 'Срединный', 'Глубокий'],
            'Массаж': ["Классический", "Лимфодренажный", "Скульптурный", 
                        "Буккальный", "Миофасциальный"],
            'Тейпирование': ["Лимфодренажное", "Лифтинговое", "Расслабляющее"],
        }

    def load_model(self):
        try:
            self._download_models()
            
            print(f"Попытка загрузки T5 модели из {self.model_path}...")
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.model_path, 
                    legacy=False, 
                    clean_up_tokenization_spaces=True
                )
                print("Токенизатор T5 успешно загружен.")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
                print("Модель ruT5-base успешно загружена.")
            except Exception as e:
                print(f"Ошибка загрузки T5 модели: {str(e)}")
                self.tokenizer = None
                self.model = None
            
            try:
                print("Попытка загрузки классификатора и регрессора...")
                models = joblib.load(self.pkl_path)
                self.classifier = models.get('best_classifier')
                self.regressor = models.get('best_regressor')
                self.vectorizer = models.get('preprocessor')
                self.label_encoders = models.get('label_encoders', {})
                print("Классификатор и регрессор успешно загружены.")
            except Exception as e:
                print(f"Ошибка загрузки классификатора и регрессора: {str(e)}")
                self.classifier = None
                self.regressor = None
                self.vectorizer = None
                self.label_encoders = None
            
            self.is_trained = True
            print("Все модели успешно загружены")
            
        except Exception as e:
            print(f"Общая ошибка загрузки моделей: {str(e)}")
            self.is_trained = False

    def _calculate_similarity(self, text1, text2):
        print("Similarity модель удалена, возвращаем 0.0")
        return 0.0

    def _generate_prompt(self, recommendation, context):
        method = recommendation['method']
        type_ = recommendation['type']
        forbidden_methods = []
        for cat, types in self.treatment_categories.items():
            if cat != method:
                forbidden_methods.extend(types)
        
        return f"""
    [Задача] Сгенерируй персонализированную рекомендацию для:
    - Тип кожи: {context['skin_type']}
    - Проблема: {context['problem']}
    - Метод: {recommendation['method']}
    - Тип процедуры: {recommendation['type']}

    [Требования]
    - Строго используй указанный метод "{method}" и тип процедуры "{type_}" в тексте. Не заменяй их другими методами или процедурами.
    - Для метода "{method}" используй только процедуры из категории "{method}". Не используй процедуры из других категорий, таких как: {', '.join(forbidden_methods)}.
    - Следуй формату:
      Инструкция: [1-2 предложения, начинающиеся с глагола]
      Противопоказания: [3 пункта, разделённые запятыми]
      Ожидаемые результаты: [2-3 эффекта, разделённые запятыми, без повторений]
    - Разделяй секции переносом строки.

    [Пример] Для "Аппаратная косметология - Ультразвуковая чистка":
    Инструкция: Пройди курс из 3-5 процедур ультразвуковой чистки у косметолога для глубокого очищения пор.
    Противопоказания: активные воспаления, купероз, онкология
    Ожидаемые результаты: уменьшение чёрных точек, гладкая текстура кожи

    [Контекст] Дополнительные симптомы: {list_to_text(context['symptoms'])}
    """

    def _validate_generation(self, generated_text, context):
        return True  # Валидация отключена

    def _postprocess_generated_text(self, generated_text, method, type_):
        for cat, types in self.treatment_categories.items():
            if cat != method:
                for t in types:
                    if t in generated_text and t != type_:
                        generated_text = generated_text.replace(t, type_)
        
        expected_results_match = re.search(r'Ожидаемые результаты\s*:?\s*(.*)$', generated_text, re.DOTALL)
        if expected_results_match:
            results = expected_results_match.group(1).strip()
            effects = [effect.strip() for effect in results.split(',')]
            unique_effects = list(dict.fromkeys(effects))[:3]
            new_results = ', '.join(unique_effects)
            generated_text = generated_text.replace(results, new_results)
        
        return generated_text

    def _parse_generated_text(self, text, method, type_, context):
        sections = {'instruction': '', 'contraindications': '', 'expected_results': ''}
        
        instruction_match = re.search(r'Инструкция\s*:?\s*(.*?)(?=\s*Противопоказания|$)', text, re.DOTALL)
        contraindications_match = re.search(r'Противопоказания\s*:?\s*(.*?)(?=\s*Ожидаемые результаты|$)', text, re.DOTALL)
        expected_results_match = re.search(r'Ожидаемые результаты\s*:?\s*(.*)$', text, re.DOTALL)
        
        if instruction_match:
            sections['instruction'] = instruction_match.group(1).strip()
        if contraindications_match:
            sections['contraindications'] = contraindications_match.group(1).strip()
        if expected_results_match:
            sections['expected_results'] = expected_results_match.group(1).strip()
        
        if not all(sections.values()):
            print(f"Парсинг не удался для {method} - {type_}, возвращаем то, что спарсили")
        
        return sections

    def get_recommendations(self, user_data):
        skin_type = user_data.get('skin_type', 'Не определён')
        symptoms = user_data.get('symptoms', [])  # Ожидаем строку или список
        effects = user_data.get('effects', [])    # Ожидаем список
        age_range = user_data.get('age_range', 'Не определён')
        problem = user_data.get('problem', 'Не определён')
        problems = problem.split(', ') if problem and problem != 'Не определён' else symptoms_to_problems(symptoms.split() if isinstance(symptoms, str) else symptoms)

        recommendations = []
        self.used_combinations = set()

        if self.classifier and self.regressor and self.label_encoders and self.vectorizer:
            try:
                for prob in problems:
                    if not prob:
                        continue
                    # Убедимся, что symptoms — это строка
                    symptoms_str = symptoms if isinstance(symptoms, str) else ' '.join(symptoms)
                    input_df = pd.DataFrame([{
                        'problem': prob,
                        'skin_type': skin_type,
                        'age_range': age_range,
                        'symptoms': [symptoms_str]  # Передаём как строку
                    }])
                    
                    print(f"Input DataFrame: {input_df}")
                    
                    processed = self.vectorizer.transform(input_df)
                    print(f"Processed data shape: {processed.shape}")
                    
                    if isinstance(self.classifier, dict):  # Для XGBoost и CatBoost
                        method_probs = self.classifier['method'].predict_proba(processed)[0]
                        type_probs = self.classifier['type'].predict_proba(processed)[0]
                        effectiveness_probs = self.classifier['effectiveness'].predict_proba(processed)[0]
                    else:  # Для остальных моделей
                        cat_pred = self.classifier.predict_proba(processed)
                        method_probs = cat_pred[0]
                        type_probs = cat_pred[1]
                        effectiveness_probs = cat_pred[2]

                    method_idx = np.argmax(method_probs)
                    method = self.label_encoders['method'].inverse_transform([method_idx])[0]
                    
                    type_idx = np.argmax(type_probs)
                    type_ = self.label_encoders['type'].inverse_transform([type_idx])[0]
                    
                    effectiveness_idx = np.argmax(effectiveness_probs)
                    effectiveness = self.label_encoders['effectiveness'].inverse_transform([effectiveness_idx])[0]
                    
                    course_duration = int(self.regressor.predict(processed)[0])
                    
                    if method == 'Массаж':
                        course_duration = min(course_duration, 30)
                    elif method == 'Уходовая косметика':
                        course_duration = min(course_duration, 30)
                    elif method == 'Аппаратная косметология':
                        course_duration = min(course_duration, 10)
                    elif method == 'Инъекционная косметология':
                        course_duration = min(course_duration, 1)
                    
                    if effectiveness in ['Низкая', 'Средняя'] and prob in self.problem_to_recommendation:
                        for rec in self.problem_to_recommendation[prob]:
                            if rec['effectiveness'] == 'Высокая':
                                method = rec['method']
                                type_ = rec['type']
                                effectiveness = rec['effectiveness']
                                course_duration = rec['course_duration']
                                break
                    
                    combo = (method, type_)
                    if combo in self.used_combinations:
                        continue
                    
                    recommendations.append({
                        'method': method,
                        'type': type_,
                        'course_duration': course_duration,
                        'effectiveness': effectiveness,
                        'problems': [prob]
                    })
                    self.used_combinations.add(combo)
            except Exception as e:
                print(f"Ошибка предсказания: {str(e)}")
                for prob in problems:
                    if prob and prob in self.problem_to_recommendation:
                        for rec in self.problem_to_recommendation[prob]:
                            rec_copy = rec.copy()
                            rec_copy['problems'] = [prob]
                            combo = (rec_copy['method'], rec_copy['type'])
                            if combo not in self.used_combinations:
                                recommendations.append(rec_copy)
                                self.used_combinations.add(combo)
        else:
            print("Классификатор или регрессор не загружены. Используем дефолтные правила.")
            for prob in problems:
                if prob and prob in self.problem_to_recommendation:
                    for rec in self.problem_to_recommendation[prob]:
                        rec_copy = rec.copy()
                        rec_copy['problems'] = [prob]
                        combo = (rec_copy['method'], rec_copy['type'])
                        if combo not in self.used_combinations:
                            recommendations.append(rec_copy)
                            self.used_combinations.add(combo)

        for recommendation in recommendations:
            context = {
                'skin_type': skin_type,
                'problem': problem,
                'symptoms': symptoms,  # Оставляем как строку для генерации
                'effects': effects,
                'type': recommendation['type']
            }
            prompt = self._generate_prompt(recommendation, context)
            
            if self.model and self.tokenizer:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    outputs = self.model.generate(
                        **inputs,
                        max_length=250,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        sep_token_id=self.tokenizer.sep_token_id
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Сгенерированный текст для {recommendation['method']} - {recommendation['type']}: {generated_text}")
                    
                    generated_text = self._postprocess_generated_text(
                        generated_text,
                        recommendation['method'],
                        recommendation['type']
                    )
                    
                    parsed_data = self._parse_generated_text(
                        generated_text,
                        recommendation['method'],
                        recommendation['type'],
                        context
                    )
                    
                    print(f"Спарсенные данные для {recommendation['method']} - {recommendation['type']}: {parsed_data}")
                    recommendation.update(parsed_data)
                except Exception as e:
                    print(f"Ошибка генерации для {recommendation['method']} - {recommendation['type']}: {str(e)}")
                    recommendation.update({
                        'instruction': '',
                        'contraindications': '',
                        'expected_results': ''
                    })
            else:
                print("Модель T5 не загружена.")
                recommendation.update({
                    'instruction': '',
                    'contraindications': '',
                    'expected_results': ''
                })

        return {'daily_routine': recommendations, 'products': [], 'procedures': []}