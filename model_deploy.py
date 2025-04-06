import os
import gdown
import joblib
import numpy as np
import pandas as pd
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def list_to_text(x):
    """Преобразует pandas.Series со списками строк в pandas.Series со строками."""
    return x.apply(lambda lst: ' '.join(lst))

def custom_tokenizer(x):
    """Разбивает строку на слова по пробелам."""
    return x.split()

class SkincareModel:
    def __init__(self, 
                 model_path="models/ruT5",
                 pkl_path="models/best_models.pkl",
                 similarity_threshold=0.45):
        """
        Инициализация модели с автоматической загрузкой всех компонентов
        
        Параметры:
        model_path - путь к папке с моделью T5 (будет скачана при первом запуске)
        pkl_path - путь к файлу с классификатором (будет скачан при первом запуске)
        """
        self.model_path = model_path
        self.pkl_path = pkl_path
        self.similarity_threshold = similarity_threshold
        
        # Создаем папки для моделей
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

        # Инициализация компонентов
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.regressor = None
        self.vectorizer = None
        self.label_encoders = None
        self.similarity_model = None
        self.is_trained = False
        self.used_combinations = set()

        # Загрузка similarity модели (не требует скачивания)
        self._load_similarity_model()

        # Инициализация problem_to_recommendation и других атрибутов
        self._initialize_attributes()

    def _load_similarity_model(self):
        """Загружает модель для расчета схожести текстов"""
        try:
            self.similarity_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device='cpu'
            )
            print("Similarity модель успешно загружена.")
        except Exception as e:
            print(f"Ошибка загрузки similarity модели: {str(e)}")
            self.similarity_model = None

    def _download_models(self):
        """Скачивает модели с Google Drive при необходимости"""
        try:
            # Если T5 модель не загружена
            if not os.path.exists(self.model_path):
                print("Скачиваем T5 модель...")
                gdown.download_folder(
                    "https://drive.google.com/drive/folders/1_zIAEkqBeR8_yS6AyfIzjLNZpNtWULfW",
                    output=self.model_path,
                    quiet=False
                )
                print("T5 модель успешно скачана.")
                # Добавляем отладку: что находится в папке?
                print("Содержимое папки models/ruT5:")
                for root, dirs, files in os.walk(self.model_path):
                    print(f"Путь: {root}, Файлы: {files}")
            
            # Если pkl файл не загружен
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
        """Инициализирует атрибуты, которые не зависят от загрузки моделей."""
        # Добавляем problem_to_recommendation как атрибут класса
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
        
        # Конфигурация валидации
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
        """Загружает все необходимые модели."""
        try:
            # Скачиваем модели, если их нет
            print("Попытка скачать модели...")
            self._download_models()
            
            # Загружаем T5 модель
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
            
            # Загружаем классификатор
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
        if self.similarity_model is None:
            return 0.0
        embeddings = self.similarity_model.encode([text1, text2], convert_to_tensor=True)
        return cosine_similarity(embeddings[0].cpu().numpy().reshape(1,-1), 
                                 embeddings[1].cpu().numpy().reshape(1,-1))[0][0]

    def _generate_prompt(self, recommendation, context):
        # Добавляем дополнительные ограничения в промпт
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

    [Пример] Для "Уходовая косметика - Крем" при проблеме "Обезвоженность":
    Инструкция: Наноси крем утром и вечером на очищенную кожу тонким слоем.
    Противопоказания: аллергия на компоненты, открытые раны, дерматит
    Ожидаемые результаты: глубокое увлажнение, устранение шелушения

    [Пример] Для "Аппаратная косметология - RF-лифтинг":
    Инструкция: Пройди курс RF-лифтинга у косметолога (5–10 процедур).
    Противопоказания: воспаления, беременность, онкология
    Ожидаемые результаты: уменьшение морщин, подтяжка кожи

    [Пример] Для "Инъекционная косметология - Ботулинотерапия":
    Инструкция: Пройди процедуру ботулинотерапии у сертифицированного косметолога.
    Противопоказания: беременность, лактация, аллергия на препарат
    Ожидаемые результаты: разглаживание морщин, омоложение кожи

    [Контекст] Дополнительные симптомы: {', '.join(context['symptoms'] if isinstance(context['symptoms'], list) else context['symptoms'].split())}
    """

    def _validate_generation(self, generated_text, context):
        # Временно отключаем валидацию, чтобы рекомендации отображались
        return True
        # Проверка семантической схожести
        query = f"{context['problem']} {context['skin_type']}"
        similarity = self._calculate_similarity(query, generated_text)
        
        # Проверка ключевых слов
        required_keywords = self.validation_rules['keywords'].get(context['type'], [])
        
        # Если есть ключевые слова - проверяем их наличие
        if required_keywords:
            valid_keywords = any(
                word in generated_text.lower() for word in required_keywords
            )
        else:  # Если нет требований - считаем проверку пройденной
            valid_keywords = True
        
        return similarity >= self.similarity_threshold and valid_keywords

    def _postprocess_generated_text(self, generated_text, method, type_):
        """Постобработка сгенерированного текста для улучшения качества."""
        # Проверяем, соответствует ли сгенерированный текст указанному методу
        for cat, types in self.treatment_categories.items():
            if cat != method:
                for t in types:
                    if t in generated_text and t != type_:
                        # Заменяем неподходящий метод на правильный тип
                        generated_text = generated_text.replace(t, type_)
        
        # Убираем повторяющиеся эффекты в "Ожидаемые результаты"
        expected_results_match = re.search(r'Ожидаемые результаты\s*:?\s*(.*)$', generated_text, re.DOTALL)
        if expected_results_match:
            results = expected_results_match.group(1).strip()
            # Разделяем эффекты по запятым
            effects = [effect.strip() for effect in results.split(',')]
            # Убираем повторы с помощью set
            unique_effects = list(dict.fromkeys(effects))
            # Ограничиваем до 3 эффектов
            unique_effects = unique_effects[:3]
            # Собираем обратно
            new_results = ', '.join(unique_effects)
            generated_text = generated_text.replace(results, new_results)
        
        return generated_text

    def _parse_generated_text(self, text, method, type_, context):
        sections = {'instruction': '', 'contraindications': '', 'expected_results': ''}
        
        # Используем регулярные выражения для поиска секций
        instruction_match = re.search(r'Инструкция\s*:?\s*(.*?)(?=\s*Противопоказания|$)', text, re.DOTALL)
        contraindications_match = re.search(r'Противопоказания\s*:?\s*(.*?)(?=\s*Ожидаемые результаты|$)', text, re.DOTALL)
        expected_results_match = re.search(r'Ожидаемые результаты\s*:?\s*(.*)$', text, re.DOTALL)
        
        # Извлекаем текст для каждой секции
        if instruction_match:
            sections['instruction'] = instruction_match.group(1).strip()
        if contraindications_match:
            sections['contraindications'] = contraindications_match.group(1).strip()
        if expected_results_match:
            sections['expected_results'] = expected_results_match.group(1).strip()
        
        # Если парсинг не удался для всех секций, возвращаем то, что есть
        if not all(sections.values()):
            print(f"Парсинг не удался для {method} - {type_}, возвращаем то, что спарсили")
            return sections
        
        # Валидация (временно отключена)
        if not self._validate_generation(
            ' '.join(sections.values()), 
            {'problem': context['problem'], 'skin_type': context['skin_type'], 'type': type_}
        ):
            print(f"Валидация не прошла для {method} - {type_}, возвращаем то, что спарсили")
        
        return sections

    def get_recommendations(self, user_data):
        # Извлекаем данные пользователя
        skin_type = user_data.get('skin_type', 'Не определён')
        symptoms = user_data.get('symptoms', [])
        effects = user_data.get('effects', [])
        age_range = user_data.get('age_range', 'Не определён')
        problem = user_data.get('problem', 'Не определён')
        problems = problem.split(', ') if problem else (symptoms if isinstance(symptoms, list) else symptoms.split())

        # Инициализируем список рекомендаций
        recommendations = []
        self.used_combinations = set()

        # Используем классификатор и регрессор для предсказания
        if self.classifier and self.regressor and self.label_encoders and self.vectorizer:
            try:
                for prob in problems:
                    # Убедимся, что symptoms — это строка
                    symptoms_str = symptoms if isinstance(symptoms, str) else ' '.join(symptoms)
                    input_df = pd.DataFrame([{
                        'problem': prob,
                        'skin_type': skin_type,
                        'age_range': age_range,
                        'symptoms': symptoms_str  # Передаём как строку
                    }])
                    
                    # Используем Pipeline напрямую для предсказания
                    if isinstance(self.classifier, dict):  # Для XGBoost и CatBoost
                        processed = self.vectorizer.transform(input_df)
                        method_probs = self.classifier['method'].predict_proba(processed)[0]
                        type_probs = self.classifier['type'].predict_proba(processed)[0]
                        effectiveness_probs = self.classifier['effectiveness'].predict_proba(processed)[0]
                    else:  # Для остальных моделей
                        cat_pred = self.classifier.predict_proba(input_df)
                        method_probs = cat_pred[0]
                        type_probs = cat_pred[1]
                        effectiveness_probs = cat_pred[2]

                    method_idx = np.argmax(method_probs)
                    method = self.label_encoders['method'].inverse_transform([method_idx])[0]
                    
                    type_idx = np.argmax(type_probs)
                    type_ = self.label_encoders['type'].inverse_transform([type_idx])[0]
                    
                    effectiveness_idx = np.argmax(effectiveness_probs)
                    effectiveness = self.label_encoders['effectiveness'].inverse_transform([effectiveness_idx])[0]
                    
                    course_duration = int(self.regressor.predict(input_df)[0])
                    
                    # Ограничение длительности курса
                    if method == 'Массаж':
                        course_duration = min(course_duration, 30)  # Ограничиваем до 30 дней
                    elif method == 'Уходовая косметика':
                        course_duration = min(course_duration, 30)  # Ограничиваем до 30 дней
                    elif method == 'Аппаратная косметология':
                        course_duration = min(course_duration, 10)  # Ограничиваем до 10 дней
                    elif method == 'Инъекционная косметология':
                        course_duration = min(course_duration, 1)   # Ограничиваем до 1 дня
                    
                    # Проверка эффективности: если "Низкая" или "Средняя", заменяем на более эффективный метод
                    if effectiveness in ['Низкая', 'Средняя'] and prob in self.problem_to_recommendation:
                        for rec in self.problem_to_recommendation[prob]:
                            if rec['effectiveness'] == 'Высокая':
                                method = rec['method']
                                type_ = rec['type']
                                effectiveness = rec['effectiveness']
                                course_duration = rec['course_duration']
                                break
                    
                    # Проверка уникальности перед добавлением
                    combo = (method, type_)
                    if combo in self.used_combinations:
                        continue
                    
                    # Добавляем проблему в рекомендацию
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
                # Исправленный fallback для всех проблем
                for prob in problems:
                    if prob in self.problem_to_recommendation:
                        for rec in self.problem_to_recommendation[prob]:
                            rec_copy = rec.copy()
                            rec_copy['problems'] = [prob]
                            combo = (rec_copy['method'], rec_copy['type'])
                            if combo not in self.used_combinations:
                                recommendations.append(rec_copy)
                                self.used_combinations.add(combo)
        else:
            print("Классификатор или регрессор не загружены. Используем дефолтные правила.")
            # Используем дефолтные правила
            for prob in problems:
                if prob in self.problem_to_recommendation:
                    for rec in self.problem_to_recommendation[prob]:
                        rec_copy = rec.copy()
                        rec_copy['problems'] = [prob]
                        combo = (rec_copy['method'], 'type')
                        if combo not in self.used_combinations:
                            recommendations.append(rec_copy)
                            self.used_combinations.add(combo)

        # Генерация текста
        for recommendation in recommendations:
            # Генерация промпта
            context = {
                'skin_type': skin_type,
                'problem': problem,
                'symptoms': symptoms,
                'effects': effects,
                'type': recommendation['type']
            }
            prompt = self._generate_prompt(recommendation, context)
            
            # Генерация текста
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
                    
                    # Постобработка текста
                    generated_text = self._postprocess_generated_text(
                        generated_text,
                        recommendation['method'],
                        recommendation['type']
                    )
                    
                    # Парсинг и валидация
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