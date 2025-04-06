print("Начало выполнения skin_app.py")
import streamlit as st
import json
import os
from datetime import datetime
import uuid
import spacy
import pandas as pd
import gdown
from PIL import Image
import base64
from utils import list_to_text, custom_tokenizer
from model_deploy import SkincareModel
from pdf import generate_pdf_report

print("Импорты завершены, создаём SkincareModel...")

# ==================== КОНСТАНТЫ ====================
SKIN_TYPES = ['Нормальная', 'Сухая', 'Жирная', 'Не уверен(а)']
AGE_RANGES = ['18-25', '25-35', '35-45', '45+']
EFFECTS = ["Увлажнение", "Лифтинг", "Устранение морщин", "Очищение пор", "Выравнивание тона",
          "Себорегуляция", "Противовоспалительный", "Антивозрастной", "Осветление", "Регенерация"]
SYMPTOMS = {
    'Чёрные точки': ['Закупоренные поры', 'Тёмные точки на поверхности кожи', 'Неровная текстура кожи', 'Шероховатость при прикосновении'],
    'Морщины': ['Мелкие линии и складки', 'Потеря упругости', '"Гусиные лапки" вокруг глаз', 'Носогубные складки'],
    'Обезвоженность': ['Чувство стянутости', 'Шелушение', 'Тусклый цвет лица', 'Повышенная чувствительность'],
    'Жирный блеск': ["Излишняя работа сальных желёз", 'Блестящая кожа в Т-зоне', 'Расширенные поры', 'Склонность к акне']
}

BACKGROUND_IMAGE_PATH = os.path.join("assets", "beauty.jpg")

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
        </style>
        """,
        unsafe_allow_html=True
    )

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return ""

# ==================== МОДЕЛИ ====================
@st.cache_resource
def load_spacy_model():
    print("Загружаем ru_core_news_sm...")
    nlp = spacy.load("ru_core_news_sm")
    print("ru_core_news_sm загружен")
    return nlp

@st.cache_resource
def load_model():
    try:
        os.makedirs("models", exist_ok=True)
        
        t5_folder = "models/ruT5"
        required_files = [
            "spiece.model", "training_args.bin", "tokenizer_config.json",
            "generation_config.json", "spiece.model.old", "config.json",
            "added_tokens.json", "special_tokens_map.json", "model.safetensors"
        ]
        all_files_present = all(os.path.exists(os.path.join(t5_folder, f)) for f in required_files)

        if not all_files_present:
            print("Скачиваем T5 модель...")
            t5_url = "https://drive.google.com/drive/folders/1_zIAEkqBeR8_yS6AyfIzjLNZpNtWULfW"
            gdown.download_folder(t5_url, output=t5_folder, quiet=False)
            print("T5 модель скачана")

        pkl_path = "models/best_models.pkl"
        if not os.path.exists(pkl_path):
            print("Скачиваем best_models.pkl...")
            pkl_url = "https://drive.google.com/uc?id=1SdvXJDBxrUS_d_Y0p5qlyU1Cw1ONc5wg"
            gdown.download(pkl_url, pkl_path, quiet=False)
            print("best_models.pkl скачан")

        model = SkincareModel(model_path=t5_folder, pkl_path=pkl_path)
        model.load_model()
        print("SkincareModel создан и загружен")
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None

# ==================== ФУНКЦИИ ====================
def symptoms_to_problems(symptoms):
    symptom_to_problem = {}
    for problem, symptom_list in SYMPTOMS.items():
        for symptom in symptom_list:
            symptom_to_problem[symptom] = problem
    return list(set(symptom_to_problem.get(s, "") for s in symptoms if s in symptom_to_problem))

def save_to_json(data):
    try:
        os.makedirs("user_data", exist_ok=True)
        filename = f"user_data/{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Ошибка сохранения: {str(e)}")
        return False

# ==================== ИНТЕРФЕЙС ====================
def init_session():
    if 'session_id' not in st.session_state:
        st.session_state.update({
            'session_id': str(uuid.uuid4()),
            'responses': {},
            'page': "questionnaire",
            'last_save': None
        })

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
        col1, col2 = st.columns(2)
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

    with st.container():
        st.markdown("### Параметры кожи 🌸")
        col3, col4 = st.columns(2)
        with col3:
            skin_type = st.radio(
                "Тип кожи*",
                options=SKIN_TYPES,
                index=None,
                key='skin_type',
                help="Выберите тип кожи, чтобы мы могли подобрать подходящие средства 🧴"
            )
        with col4:
            all_symptoms = [s for group in SYMPTOMS.values() for s in group]
            symptoms = st.multiselect(
                "Беспокоящие симптомы*",
                options=all_symptoms,
                placeholder="Выберите симптомы...",
                key='symptoms',
                help="Расскажите, что вас беспокоит, чтобы мы могли помочь 💖"
            )
        
        effects = st.multiselect(
            "Ожидаемые эффекты*",
            options=EFFECTS,
            placeholder="Выберите желаемые эффекты...",
            key='effects',
            help="Какие результаты вы хотите получить? ✨"
        )

    if st.button(
        "Получить рекомендации", 
        type="primary", 
        use_container_width=True,
        key="main_button"
    ):
        required_fields = {
            'name': bool(name),
            'skin_type': skin_type in SKIN_TYPES[:3],
            'age_range': bool(age_range),
            'symptoms': len(symptoms) > 0,
            'effects': len(effects) > 0
        }
        
        if all(required_fields.values()):
            st.session_state.responses = {
                'name': name,
                'age_range': age_range,
                'skin_type': skin_type,
                'symptoms': symptoms,  # Оставляем список
                'effects': effects,    # Оставляем список
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
    
    if 'responses' not in st.session_state:
        st.error("Данные не найдены! Заполните анкету сначала.")
        st.stop()
    
    st.progress(100)
    st.markdown(f"<p style='text-align: center; color: #FF9999; font-weight: 500;'>Готово, {st.session_state.responses.get('name', 'Друг')}!</p>", unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(BACKGROUND_IMAGE_PATH)
            st.image(image, use_column_width=True)
        with col2:
            st.markdown("""
            <div style="padding: 1rem;">
                <h3>Персонализированные рекомендации</h3>
                <p>Советы по уходу, подобранные специально для вас</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Не удалось загрузить изображение: {str(e)}")
    
    model = load_model()
    if model is None:
        st.error("Не удалось загрузить модель рекомендаций")
        return
    
    with st.spinner("Формируем рекомендации..."):
        try:
            user_data = st.session_state.responses.copy()
            # Передаём данные как есть, Pipeline сам обработает
            recommendations = model.get_recommendations(user_data)
            st.session_state.recommendations = recommendations
            
            st.markdown("### Ваш профиль")
            st.markdown(f"""
            - **Имя:** {st.session_state.responses.get('name', 'Не указано')}
            - **Возраст:** {st.session_state.responses.get('age_range', 'Не указан')}
            - **Тип кожи:** {st.session_state.responses.get('skin_type', 'Не определён')}
            - **Основные проблемы:** {list_to_text(st.session_state.responses.get('symptoms', [])) or 'Не указаны'}
            """)
            
            if recommendations and 'daily_routine' in recommendations:
                st.markdown("### Рекомендации по уходу")
                for item in recommendations['daily_routine']:
                    with st.expander(f"💡 {item.get('method', 'Метод')} - {item.get('type', 'Тип')}", expanded=True):
                        st.markdown(f"""
                        - **Инструкция:** {item.get('instruction', '')}
                        - **Противопоказания:** {item.get('contraindications', '')}
                        - **Ожидаемые результаты:** {item.get('expected_results', '')}
                        - **Эффективность:** {item.get('effectiveness', '')}
                        - **Длительность курса:** {item.get('course_duration', '')} дней
                        """)
            else:
                st.info("Рекомендации не найдены")
                
        except Exception as e:
            st.error(f"Ошибка при формировании рекомендаций: {str(e)}")
            return
    
    if st.button("Сохранить отчёт в PDF", use_container_width=True):
        try:
            user_data = st.session_state.responses.copy()
            user_data['session_id'] = st.session_state.session_id
            pdf_path = generate_pdf_report(user_data, st.session_state.recommendations)
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label="Скачать PDF",
                data=pdf_data,
                file_name=f"skincare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Ошибка при создании PDF: {str(e)}")
    
    if st.button("Новый анализ", use_container_width=True):
        st.session_state.page = "questionnaire"
        st.rerun()

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    print("Начало основного блока")
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    set_custom_style()
    print("Стили установлены")
    nlp = load_spacy_model()
    print("Spacy загружен")
    
    if st.session_state.get('page', 'questionnaire') == "questionnaire":
        main_questionnaire()
    else:
        show_recommendations()
    
    print("Конец основного блока")
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 1rem;'>
            <p style="font-size: 0.9em;">© 2025 Beauty Tracker | Создано с заботой для вашей кожи 🌸</p>
        </div>
        """,
        unsafe_allow_html=True
    )