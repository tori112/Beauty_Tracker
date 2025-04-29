import streamlit as st
import json
import os
import logging
from datetime import datetime
import uuid
import pandas as pd
from PIL import Image
import base64
from mod import predict_for_multiple_problems  # Основная функция
from pdf import generate_pdf_report
import sys

# Настройка логирования с кодировкой UTF-8
os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", "app.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []

# Обработчик для файла с кодировкой UTF-8
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Обработчик для консоли с кодировкой UTF-8
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if sys.stdout.encoding != 'utf-8':
    try:
        console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    except Exception as e:
        logging.error(f"Failed to set console encoding to UTF-8: {str(e)}")
logger.addHandler(console_handler)

# Функция list_to_text
def list_to_text(x):
    """Преобразует список симптомов или Series в строку, разделённую пробелами"""
    if isinstance(x, pd.Series):
        return x.apply(lambda lst: ' '.join(lst))
    return ' '.join(x)  # Для обычного списка

# ==================== КОНСТАНТЫ ====================
MODEL_DIR = r"D:\MIPT\IDE\knowledge_project_backup\skincare_project\deploy\models"
TEMPLATES_PATH = os.path.join(MODEL_DIR, "valid_templates.json")

SKIN_TYPES = ['Нормальная', 'Сухая', 'Жирная', 'Не уверен(а)']
AGE_RANGES = ['18-25', '25-35', '35-45', '45+']
GENDERS = ['Мужской', 'Женский']
EFFECTS = ["Увлажнение", "Лифтинг", "Устранение морщин", "Очищение пор", "Выравнивание тона",
          "Устранение жирности кожи лица", "Противовоспалительный", "Антивозрастной", "Осветление", "Восстановление кожи лица"]
SYMPTOMS = {
    'Черные точки': ['Закупоренные поры', 'Темные точки на поверхности кожи', 'Неровная текстура кожи', 'Шероховатость при прикосновении', 'Серый цвет лица из-за окисления кожного сала'],
    'Морщины': ['Мелкие линии и складки на коже', 'Потеря упругости',
                'Дряблость кожи', '"Гусиные лапки" вокруг глаз', 'Носогубные складки',
                'Неравномерный рельеф кожи', 'Потеря четкости овала лица'],
    'Обезвоженность': ['Чувство стянутости', 'Шелушение',
                       'Тусклый цвет лица', 'Мелкие морщинки',
                       'Повышенная чувствительность', 'Раздражение', 'Зуд', 'Неравномерный тон',
                       'При надавливании кожа медленно возвращается в исходное положение'],
    'Жирный блеск': ["Излишняя работа сальных желез",
                     'Блестящая кожа, особенно в Т-зоне',
                     'Расширенные поры', 'Склонность к образованию акне',
                     'Жирность появляется через 2-3 часа после умывания',
                     'Макияж быстро "плывет"']
}
SYMPTOM_DESCRIPTIONS = {
    'Закупоренные поры': 'Поры забиты кожным салом или омертвевшими клетками, что может привести к появлению черных точек или акне.',
    'Темные точки на поверхности кожи': 'Маленькие темные пятна, обычно на носу, щеках или подбородке, вызванные окислением кожного сала в порах.',
    'Неровная текстура кожи': 'Кожа выглядит грубой или неровной на ощупь, часто из-за сухости или закупоренных пор.',
    'Шероховатость при прикосновении': 'Кожа ощущается шершавой или грубой при касании, часто из-за сухости или ороговения.',
    'Мелкие линии и складки': 'Тонкие морщины, которые появляются на коже, часто из-за старения или сухости.',
    'Потеря упругости': 'Кожа становится менее эластичной, может казаться дряблой или обвисшей.',
    '"Гусиные лапки" вокруг глаз': 'Мелкие морщинки, расходящиеся от внешних уголков глаз, напоминающие следы лапок.',
    'Носогубные складки': 'Глубокие линии, идущие от крыльев носа к уголкам рта, часто усиливаются с возрастом.',
    'Чувство стянутости': 'Ощущение, что кожа "натянута" или сухая, особенно после умывания.',
    'Шелушение': 'Кожа отслаивается мелкими чешуйками, часто из-за сухости или обезвоживания.',
    'Тусклый цвет лица': 'Кожа выглядит бледной, сероватой или лишенной сияния.',
    'Повышенная чувствительность': 'Кожа реагирует покраснением, жжением или раздражением на косметику, погоду и т.д.',
    'Излишняя работа сальных желез': 'Кожа становится жирной из-за чрезмерного выделения кожного сала.',
    'Блестящая кожа в Т-зоне': 'Лоб, нос и подбородок выглядят жирными и блестящими, особенно к середине дня.',
    'Расширенные поры': 'Поры выглядят крупными и заметными, часто на носу, щеках или подбородке.',
    'Склонность к акне': 'Частое появление прыщей, воспалений или угрей на коже.'
}
CONTRAINDICATIONS = [
    "Злокачественные новообразования",
    "Острые инфекционные заболевания",
    "Дерматиты или экзема",
    "Нет"
]
ALLERGIES = [
    "Аллергия на салициловую кислоту",
    "Аллергия на ретинол",
    "Аллергия на витамин С",
    "Аллергия на пептиды",
    "Аллергия на гиалуроновую кислоту",
    "Нет"
]
BACKGROUND_IMAGE_PATH = os.path.join("assets", "beauty.jpg")
SUCCESS_PROB_THRESHOLD = 0.1  # Порог эффективности (10%)
CARE_PRODUCT_TYPES = ["Крем", "Сыворотка", "Гель", "Тоник", "Маска", "Патчи", "Мыло"]

# ==================== СТИЛИ ====================
def set_custom_style():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 245, 245, 0.9), rgba(255, 245, 245, 0.95)), 
                       url('data:image/jpg;base64,{image_to_base64(BACKGROUND_IMAGE_PATH)}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #333333 !important;
            font-family: 'Arial', sans-serif;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 1rem;
            border: 1px solid #FF9999;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2D3436 !important;
            font-weight: 600 !important;
        }}
        body, p, label, .stTextInput>label, .stSelectbox>label, 
        .stMultiselect>label, .stRadio>label, .stCheckbox>label {{
            color: #333333 !important;
            font-weight: 500 !important;
            font-size: 1.1em !important;
        }}
        .stTextInput>div>div>input {{
            background-color: #FFFFFF !important;
            border: 1px solid #FF9999 !important;
            border-radius: 10px !important;
            padding: 10px 15px !important;
            color: #333333 !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        .stSelectbox>div>div, .stMultiSelect>div>div {{
            background-color: #FFFFFF !important;
            border: 1px solid #FF9999 !important;
            border-radius: 10px !important;
            color: #333333 !important;
        }}
        .stSelectbox>div>div>select, .stMultiSelect>div>div>ul {{
            background-color: #FFFFFF !important;
            color: #333333 !important;
        }}
        .stSelectbox>div>div>select>option, .stMultiSelect>div>div>ul>li {{
            background-color: #FFFFFF !important;
            color: #333333 !important;
        }}
        .stSelectbox>div>div>select>option:hover, .stMultiSelect>div>div>ul>li:hover {{
            background-color: #FF9999 !important;
            color: #FFFFFF !important;
        }}
        .stMultiSelect [data-baseweb="select"]>div {{
            background-color: #FFFFFF !important;
            color: #333333 !important;
        }}
        .stMultiSelect [data-baseweb="select"]>div>div {{
            background-color: #FFFFFF !important;
            color: #333333 !important;
        }}
        .stMultiSelect [data-baseweb="select"]>div>span {{
            color: #333333 !important;
        }}
        .stMultiSelect [data-baseweb="select"] ul {{
            background-color: #FFFFFF !important;
        }}
        .stMultiSelect [data-baseweb="select"] ul li {{
            background-color: #FFFFFF !important;
            color: #333333 !important;
        }}
        .stMultiSelect [data-baseweb="select"] ul li:hover {{
            background-color: #FF9999 !important;
            color: #FFFFFF !important;
        }}
        .stRadio>label {{
            color: #333333 !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            padding: 8px 12px !important;
            border-radius: 8px !important;
            margin-bottom: 8px !important;
        }}
        .stButton>button {{
            background-color: #6C5CE7 !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: transform 0.2s ease-in-out;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            background-color: #5A4BCE !important;
        }}
        .stDownloadButton>button {{
            background-color: #FF9999 !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: transform 0.2s ease-in-out;
        }}
        .stDownloadButton>button:hover {{
            transform: scale(1.05);
            background-color: #FF8080 !important;
        }}
        .stProgress>div>div>div {{
            background-color: #FF9999 !important;
        }}
        .stForm {{
            background-color: transparent !important;
        }}
        .recommendation-text p {{
            margin: 0.5em 0;
            line-height: 1.5;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Ошибка загрузки изображения {image_path}: {str(e)}")
        return ""

# ==================== ФУНКЦИИ ЛОГИРОВАНИЯ ====================
def log_user_choice(user_data, selected_recommendation):
    """Сохраняет выбор пользователя в файл"""
    try:
        log_entry = {
            'session_id': st.session_state.session_id,
            'user_data': user_data,
            'selected_recommendation': selected_recommendation,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        os.makedirs("user_choices", exist_ok=True)
        with open(f"user_choices/choice_{st.session_state.session_id}_{log_entry['timestamp']}.json", 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        logging.info(f"Выбор пользователя сохранён: {selected_recommendation['method']}_{selected_recommendation['type']}")
    except Exception as e:
        logging.error(f"Ошибка сохранения выбора: {str(e)}")
        st.error(f"Ошибка сохранения выбора: {str(e)}")

def log_user_feedback(selected_recommendation, rating, feedback):
    """Сохраняет отзыв пользователя в файл"""
    try:
        feedback_entry = {
            'session_id': st.session_state.session_id,
            'selected_recommendation': selected_recommendation,
            'rating': rating,
            'feedback': feedback,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        os.makedirs("user_feedback", exist_ok=True)
        with open(f"user_feedback/feedback_{st.session_state.session_id}_{feedback_entry['timestamp']}.json", 'w', encoding='utf-8') as f:
            json.dump(feedback_entry, f, ensure_ascii=False, indent=2)
        logging.info(f"Отзыв пользователя сохранён: рейтинг {rating}")
    except Exception as e:
        logging.error(f"Ошибка сохранения отзыва: {str(e)}")
        st.error(f"Ошибка сохранения отзыва: {str(e)}")

# ==================== ФУНКЦИИ ====================
def symptoms_to_problems(symptoms):
    symptom_to_problem = {}
    for problem, symptom_list in SYMPTOMS.items():
        unified_problem = problem.replace("ё", "е")
        for symptom in symptom_list:
            symptom_to_problem[symptom] = unified_problem
    problems = list(set(symptom_to_problem.get(s, "") for s in symptoms if s in symptom_to_problem))
    logging.info(f"Симптомы {symptoms} преобразованы в проблемы: {problems}")
    return problems

def save_to_json(data):
    try:
        os.makedirs("user_data", exist_ok=True)
        filename = f"user_data/{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Данные сохранены в {filename}")
        return True
    except Exception as e:
        logging.error(f"Ошибка сохранения данных: {str(e)}")
        st.error(f"Ошибка сохранения: {str(e)}")
        return False

def extract_expected_effects(template):
    for line in template.split('\n'):
        if "Ожидаемые эффекты:" in line:
            effects = line.split("Ожидаемые эффекты:")[1].strip().rstrip('.')
            return effects.split(", ")
    return []

def check_compatibility(template, user_allergies, is_pregnant):
    active_ingredients = template.get("active_ingredients", [])
    for allergy in user_allergies:
        allergen = allergy.replace("Аллергия на ", "").lower()
        for ingredient in active_ingredients:
            if ingredient.lower() == allergen:
                logging.info(f"Рекомендация отклонена: содержит {ingredient}, на который у пользователя аллергия")
                return False, f"Рекомендация содержит {ingredient}, на который у вас аллергия."
    if is_pregnant and template.get("contraindicated_during_pregnancy", False):
        logging.info("Рекомендация отклонена: противопоказана при беременности")
        return False, "Рекомендация противопоказана при беременности."
    return True, None

def format_template_text(template):
    lines = template.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            if ': ' in line:
                key, value = line.split(': ', 1)
                formatted_lines.append(f"**{key}:** {value}")
            else:
                formatted_lines.append(f"{line}")
    return '\n\n'.join(formatted_lines)  # Двойной перенос для разделения блоков

# ==================== ИНТЕРФЕЙС ====================
def init_session():
    if 'session_id' not in st.session_state:
        st.session_state.update({
            'session_id': str(uuid.uuid4()),
            'responses': {},
            'page': "questionnaire",
            'last_save': None,
            'confirmed_recommendation': None
        })
    logging.info(f"Инициализирована сессия с ID: {st.session_state.session_id}")

def main_questionnaire():
    init_session()
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2D3436;">Персональный подбор ухода 💖</h1>
        <p style="color: #666666;">Ответьте на вопросы для персонализированных рекомендаций</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("### Личные данные 👤")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(
                "Ваше имя*",
                placeholder="Введите ваше имя",
                key='name',
                help="Мы используем ваше имя, чтобы сделать рекомендации более персональными 💕"
            )
            if name:
                st.markdown(f"<p style='color: #FF9999; font-weight: 500;'>Привет, {name}! Давай подберём уход для тебя! 💕</p>", unsafe_allow_html=True)
        with col2:
            age_range = st.selectbox(
                "Возрастной диапазон*",
                options=AGE_RANGES,
                index=None,
                placeholder="Выберите возраст",
                key='age_range',
                help="Это поможет подобрать уход, подходящий для вашего возраста 🌟"
            )
        with col3:
            gender = st.selectbox(
                "Пол*",
                options=GENDERS,
                index=None,
                placeholder="Выберите пол",
                key='gender',
                help="Это поможет учесть особенности ухода 🌸"
            )

    is_pregnant = False
    if gender == "Женский":
        with st.container():
            st.markdown("### Дополнительные данные")
            is_pregnant = st.checkbox(
                "Я беременна",
                key='is_pregnant',
                help="Это важно для исключения процедур, которые могут быть небезопасны во время беременности 🤰"
            )

    with st.container():
        st.markdown("### Параметры кожи 🌸")
        col4, col5 = st.columns(2)
        with col4:
            skin_type = st.radio(
                "Тип кожи*",
                options=SKIN_TYPES,
                index=None,
                key='skin_type',
                help="Выберите тип кожи, чтобы мы могли подобрать подходящие средства 🧴"
            )
        
        with col5:
            st.markdown("#### Беспокоящие симптомы")
            st.markdown("Ниже вы можете ознакомиться с описанием симптомов. Нажмите на название, чтобы узнать подробности:")
            
            for problem, symptom_list in SYMPTOMS.items():
                with st.expander(f"📋 Симптомы, связанные с проблемой: {problem}", expanded=False):
                    for symptom in symptom_list:
                        description = SYMPTOM_DESCRIPTIONS.get(symptom, "Описание отсутствует.")
                        st.markdown(f"""
                        **{symptom}**  
                        {description}  
                        """)
            
            all_symptoms = [s for group in SYMPTOMS.values() for s in group]
            symptoms = st.multiselect(
                "Выберите симптомы*",
                options=all_symptoms,
                placeholder="Выберите симптомы...",
                key='symptoms',
                help="Расскажите, что вас беспокоит, чтобы мы могли помочь 💖."
            )

        effects = st.multiselect(
            "Ожидаемые эффекты*",
            options=EFFECTS,
            placeholder="Выберите желаемые эффекты...",
            key='effects',
            help="Какие результаты вы хотите получить? ✨"
        )

        contraindications = st.multiselect(
            "Противопоказания (выберите все, что применимо)",
            options=CONTRAINDICATIONS,
            placeholder="Выберите противопоказания...",
            key='contraindications'
        )

        allergies = st.multiselect(
            "Аллергии (выберите все, что применимо)",
            options=ALLERGIES,
            placeholder="Выберите аллергии...",
            key='allergies',
            help="Укажите, если у вас есть аллергия на определённые компоненты 🚫"
        )

    if st.button(
        "Получить рекомендации", 
        type="primary", 
        use_container_width=True,
        key="main_button"
    ):
        required_fields = {
            'name': bool(name),
            'gender': bool(gender),
            'skin_type': skin_type in SKIN_TYPES[:3],
            'age_range': bool(age_range),
            'symptoms': len(symptoms) > 0,
            'effects': len(effects) > 0
        }
        
        if all(required_fields.values()):
            st.session_state.responses = {
                'name': name,
                'gender': gender,
                'is_pregnant': is_pregnant,
                'age_range': age_range,
                'skin_type': skin_type,
                'symptoms': symptoms,
                'effects': effects,
                'contraindications': contraindications,
                'allergies': allergies,
                'problem': ', '.join(symptoms_to_problems(symptoms))
            }
            if save_to_json(st.session_state.responses):
                st.session_state.page = "recommendations"
                st.success(f"Отлично, {name}! Ты на пути к идеальной коже! ✨")
                st.rerun()
        else:
            missing = [k for k, v in required_fields.items() if not v]
            st.error(f"Заполните обязательные поля: {', '.join(missing)}")

def show_recommendations():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2D3436;">Ваши рекомендации 💖</h1>
        <p style="color: #666666;">Персональный план ухода за кожей</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Добавляем отладочный вывод, чтобы проверить состояние сессии
    logging.info(f"Состояние сессии в show_recommendations: {st.session_state}")
    if 'responses' not in st.session_state:
        st.error("Данные не найдены! Заполните анкету сначала.")
        logging.error("st.session_state.responses отсутствует!")
        return
    
    st.progress(100)
    st.markdown(f"<p style='text-align: center; color: #FF9999; font-weight: 500;'>Готово, {st.session_state.responses.get('name', 'Друг')}!</p>", unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(BACKGROUND_IMAGE_PATH)
            st.image(image, use_column_width=True)
        with col2:
            pass
    except Exception as e:
        st.warning(f"Не удалось загрузить изображение: {str(e)}")
    
    with st.spinner("Формируем рекомендации..."):
        user_data = st.session_state.responses
        skin_type = user_data['skin_type']
        age_range = user_data['age_range']
        user_symptoms = user_data['symptoms']
        contraindications = user_data['contraindications']
        allergies = user_data['allergies']
        is_pregnant = user_data['is_pregnant']

        shown_warning = False
        if "Нет" not in contraindications and len(contraindications) > 0:
            st.warning("⚠️ У вас есть противопоказания. Некоторые рекомендации могут быть ограничены.")
            shown_warning = True
        if "Нет" not in allergies and len(allergies) > 0 and not shown_warning:
            st.warning("⚠️ У вас указаны аллергии. Мы исключили опасные компоненты.")
            shown_warning = True
        if is_pregnant and not shown_warning:
            st.warning("⚠️ Вы беременны. Процедуры с противопоказаниями исключены.")

        user_problems = list(set(symptoms_to_problems(user_symptoms)))
        logging.info(f"Обнаружены проблемы: {user_problems}")

        all_results = predict_for_multiple_problems(
            problems=user_problems,
            skin_type=skin_type,
            age_range=age_range,
            symptoms=user_symptoms,
            user_allergies=user_data['allergies'],
            user_contraindications=user_data['contraindications'],
            is_pregnant=user_data['is_pregnant'],
            top_per_problem=3
        )

        all_recommendations = []
        problems_with_errors = []

        for problem_result in all_results:
            if 'error' in problem_result:
                error_message = problem_result['error']
                logging.warning(f"Ошибка для проблемы: {error_message}")
                problems_with_errors.append(error_message)
                continue
            
            problem = problem_result['problem']
            recommendations = problem_result.get('recommendations', [])
            
            if not recommendations:
                logging.warning(f"Для проблемы '{problem}' не найдено рекомендаций")
                continue
            
            for rec in recommendations:
                all_recommendations.append({
                    'problem': problem,
                    'symptom': ', '.join([s for s in user_symptoms if symptoms_to_problems([s])[0] == problem]),
                    'method': rec['method'],
                    'type': rec['type'],
                    'effectiveness': f"{rec['success_prob']}%",
                    'course_duration': rec.get('course_duration', 'Не указана'),
                    'expected_results': rec.get('expected_effect', 'Не указан'),
                    'contraindications': ', '.join(rec.get('contraindications', ['Нет'])),
                    'template': rec.get('template', 'Описание отсутствует')
                })

        problems_without_recs = [
            p for p in user_problems 
            if p not in {r['problem'] for r in all_recommendations}
        ]
        if problems_without_recs:
            st.info(f"ℹ️ Для проблем: {', '.join(problems_without_recs)} рекомендации не найдены для вашего типа кожи ({skin_type}) и возраста ({age_range}). Попробуйте уточнить симптомы или обратитесь к специалисту.")
        
        st.session_state.recommendations = {
            'daily_routine': all_recommendations,
            'products': [],
            'procedures': []
        }

        # Отображаем профиль пользователя (используем только Markdown)
        st.markdown("### Ваш профиль")
        allergies_text = ', '.join(st.session_state.responses.get('allergies', [])) or 'Не указаны'
        contraindications_text = ', '.join(st.session_state.responses.get('contraindications', [])) or 'Нет'
        profile_text = f"""
**Имя:** {st.session_state.responses.get('name', 'Не указано')}  
**Тип кожи:** {st.session_state.responses.get('skin_type', 'Не определён')}  
**Возраст:** {st.session_state.responses.get('age_range', 'Не указана')}  
**Основные проблемы:**  
"""
        problems = st.session_state.responses.get('problem', 'Не указаны').split(', ')
        for problem in problems:
            profile_text += f"- {problem}\n"
        profile_text += f"""
**Аллергии:** {allergies_text}  
**Противопоказания:** {contraindications_text}  
**Беременность:** {'Да' if st.session_state.responses.get('is_pregnant', False) else 'Нет'}
"""
        st.markdown(profile_text)

        # Отображаем рекомендации (используем только Markdown, разделяем блоки)
        st.markdown("### Рекомендации по уходу")
        if not all_recommendations:
            st.error("❌ Не удалось сформировать рекомендации. Проверьте наличие подходящих шаблонов в valid_templates.json и их структуру (обязательные поля: method, type).")
        else:
            for rec in all_recommendations:
                with st.expander(f"💡 {rec['problem']} ({rec['method']} - {rec['type']}) ⭐ {rec['effectiveness']}", expanded=True):
                    formatted_template = format_template_text(rec['template'])
                    st.markdown(f"""
**Симптом:** {rec['symptom']}  
**Эффективность:** {rec['effectiveness']}  
**Курс:** {rec['course_duration']}  
**Ожидаемые результаты:** {rec['expected_results']}  
**Описание:**  
{formatted_template}
                    """)

            # Добавляем выбор процедуры
            st.markdown("#### Выберите процедуру")
            recommendation_options = [
                f"{rec['problem']} ({rec['method']} - {rec['type']}) - Эффективность: {rec['effectiveness']}"
                for rec in all_recommendations
            ]
            selected_recommendation = st.selectbox(
                "Выберите процедуру, которую хотите попробовать:",
                options=["Выберите..."] + recommendation_options,
                key='selected_recommendation'
            )

            if selected_recommendation != "Выберите..." and st.button("Подтвердить выбор"):
                selected_idx = recommendation_options.index(selected_recommendation)
                selected_rec = all_recommendations[selected_idx]
                st.session_state.confirmed_recommendation = selected_rec
                log_user_choice(st.session_state.responses, selected_rec)
                st.success(f"Вы выбрали: {selected_recommendation}")

            # Добавляем форму обратной связи
            if 'confirmed_recommendation' in st.session_state and st.session_state.confirmed_recommendation:
                st.markdown("### Оцените выбранную процедуру")
                selected_rec = st.session_state.confirmed_recommendation
                st.markdown(f"**Выбранная процедура:** {selected_rec['problem']} ({selected_rec['method']} - {selected_rec['type']})")
                rating = st.slider(
                    "Как вы оцениваете эффективность процедуры? (1 - не помогло, 5 - отлично)",
                    1, 5, 3,
                    key='rating'
                )
                feedback = st.text_area(
                    "Комментарий (опционально):",
                    placeholder="Расскажите, что вам понравилось или не понравилось",
                    key='feedback'
                )
                if st.button("Отправить отзыв"):
                    log_user_feedback(selected_rec, rating, feedback)
                    st.success("Спасибо за ваш отзыв! 💖")
                    st.session_state.confirmed_recommendation = None
                    st.rerun()

        st.markdown("""
        <div style='margin-top: 2rem; padding: 1rem; background-color: rgba(255, 245, 245, 0.9); border-left: 4px solid #FF9999; border-radius: 8px;'>
            <p style='color: #FF9999; font-weight: bold;'>*Эти рекомендации носят общий характер и не заменяют профессиональную консультацию косметолога. Перед применением процедур обязательно проконсультируйтесь со специалистом, особенно если у вас есть хронические заболевания или индивидуальные особенности кожи.</p>
        </div>
        """, unsafe_allow_html=True)

    # Кнопки "Сохранить отчёт в PDF" и "Новый анализ"
    if st.button("Сохранить отчёт в PDF", use_container_width=True):
        try:
            pdf_path = generate_pdf_report(
                st.session_state.responses,
                st.session_state.recommendations,
                session_id=st.session_state.session_id
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Скачать PDF",
                    f,
                    file_name=f"skincare_report_{datetime.now().strftime('%Y%m%d')}.pdf"
                )
        except Exception as e:
            st.error(f"Ошибка при создании PDF: {e}")

    if st.button("Новый анализ", use_container_width=True):
        st.session_state.clear()
        st.session_state.page = "questionnaire"
        st.rerun()

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    set_custom_style()
    
    if st.session_state.get('page', 'questionnaire') == "questionnaire":
        main_questionnaire()
    else:
        show_recommendations()
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 1rem;'>
            <p style="font-size: 0.9em;">© 2025 Beauty Tracker | Создано с заботой для вашей кожи 🌸</p>
        </div>
        """,
        unsafe_allow_html=True
    )