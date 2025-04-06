import os
import gdown
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

# 1. Настройка шрифтов
def setup_fonts():
    # Создаем папку для шрифтов
    font_dir = "fonts"
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "DejaVuSans.ttf")
    
    # Скачиваем шрифт, если его нет
    if not os.path.exists(font_path):
        try:
            gdown.download(
                "https://drive.google.com/uc?id=1xj8bQsSs2gGiUiCdTrwy6T91q-MUo4xK",
                font_path,
                quiet=False
            )
        except Exception as e:
            print(f"Ошибка загрузки шрифта: {e}")
    
    # Регистрируем шрифт
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            return 'DejaVuSans'
    except:
        pass
    
    # Fallback на стандартный шрифт Helvetica
    return 'Helvetica'

# Инициализация шрифтов
FONT_NAME = setup_fonts()

def generate_pdf_report(user_data, recommendations):
    # Создаем папку для отчетов
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Генерируем имя файла с timestamp
    filename = os.path.join(
        reports_dir,
        f"{user_data['session_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    
    # Инициализация документа
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        encoding='utf-8'  # Важно для кириллицы
    )
    
    # Настройка стилей
    styles = getSampleStyleSheet()
    
    # Применяем шрифт ко всем стилям
    for style in ['Title', 'Heading2', 'Normal', 'Bullet']:
        styles[style].fontName = FONT_NAME
        styles[style].encoding = 'UTF-8'
    
    # Формируем содержимое PDF
    story = []
    
    # Заголовок
    title = Paragraph(
        "Персональные рекомендации по уходу за кожей",
        styles['Title']
    )
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Информация о пользователе
    user_info = f"""
    <b>Имя:</b> {user_data.get('name', 'Не указано')}<br/>
    <b>Тип кожи:</b> {user_data.get('skin_type', 'Не определён')}<br/>
    <b>Возрастная группа:</b> {user_data.get('age_range', 'Не указана')}
    """
    story.append(Paragraph(user_info, styles['Normal']))
    story.append(Spacer(1, 24))
    
    # Рекомендации
    rec_title = Paragraph(
        "<b>Ваш персональный план ухода:</b>",
        styles['Heading2']
    )
    story.append(rec_title)
    
    # Обработка рекомендаций
    all_recommendations = []
    for category in recommendations.values():
        if isinstance(category, list):
            all_recommendations.extend(category)
    
    if all_recommendations:
        for item in all_recommendations:
            # Формируем текст рекомендации
            symptoms_text = ", ".join(user_data.get('symptoms', [])) if user_data.get('symptoms') else "проблем не обнаружено"
            
            user_text = f"""
            У тебя {user_data.get('skin_type', 'не определён').lower()} кожа.
            {"Мы заметили у тебя небольшую особенность — " + symptoms_text.lower() + "." 
             if user_data.get('symptoms') 
             else "Проблем не обнаружено! Вы молодец, что так замечательно ухаживаете за кожей лица!"}
            Рекомендуем попробовать {item.get('method', 'Уходовая косметика')} 
            {item.get('type', 'Крем')} в течение {item.get('course_duration', 30)} дней.
            """
            
            story.append(Paragraph(user_text, styles['Normal']))
            
            # Детали рекомендации
            details = [
                f"Инструкция: {item.get('instruction', 'Следуйте рекомендациям специалиста')}",
                f"Эффективность: {item.get('effectiveness', 'Средняя')}"
            ]
            
            for detail in details:
                story.append(Paragraph(detail, styles['Bullet']))
            
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("Рекомендаций пока нет.", styles['Normal']))
    
    # Собираем PDF
    doc.build(story)
    return filename