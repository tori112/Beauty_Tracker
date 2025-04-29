import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from datetime import datetime

def setup_fonts():
    font_dir = "fonts"
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "DejaVuSans.ttf")
    
    if not os.path.exists(font_path):
        try:
            import urllib.request
            url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            print(f"Ошибка загрузки шрифта: {e}")
    
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            return 'DejaVuSans'
    except:
        pass
    return 'Helvetica'

FONT_NAME = setup_fonts()

def generate_pdf_report(user_data, recommendations, session_id):
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    filename = os.path.join(
        reports_dir,
        f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        encoding='utf-8'
    )
    
    styles = getSampleStyleSheet()
    for style in ['Title', 'Heading2', 'Normal', 'Bullet']:
        styles[style].fontName = FONT_NAME
        styles[style].encoding = 'UTF-8'
    
    # Настраиваем стиль для основного текста
    styles['Normal'].textColor = HexColor('#333333')
    styles['Normal'].leading = 14  # Увеличиваем межстрочный интервал
    
    # Стиль для заголовков рекомендаций
    styles.add(ParagraphStyle(
        name='RecommendationTitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        textColor=HexColor('#4A4A4A')
    ))
    
    # Стиль для сноски
    footnote_style = styles['Normal'].clone('Footnote')
    footnote_style.fontSize = 9
    footnote_style.textColor = HexColor('#666666')
    footnote_style.leading = 12
    
    story = []
    
    # Заголовок документа
    title = Paragraph("Персональные рекомендации по уходу за кожей", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Профиль пользователя (логические блоки)
    story.append(Paragraph("<b>Ваш профиль:</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    profile_items = [
        f"<b>Имя:</b> {user_data.get('name', 'Не указано')}",
        f"<b>Пол:</b> {user_data.get('gender', 'Не указан')}",
        f"<b>Беременность:</b> {'Да' if user_data.get('is_pregnant', False) else 'Нет'}",
        f"<b>Тип кожи:</b> {user_data.get('skin_type', 'Не определён')}",
        f"<b>Возрастная группа:</b> {user_data.get('age_range', 'Не указана')}",
        f"<b>Проблемы:</b>"
    ]
    
    # Добавляем каждый элемент профиля как отдельный параграф
    for item in profile_items[:-1]:  # Без проблем
        story.append(Paragraph(item, styles['Normal']))
        story.append(Spacer(1, 4))
    
    # Проблемы как список
    story.append(Paragraph(profile_items[-1], styles['Normal']))
    problems = user_data.get('problem', 'Не указаны').split(', ')
    for problem in problems:
        story.append(Paragraph(f"• {problem}", styles['Bullet']))
    
    # Аллергии
    story.append(Paragraph(f"<b>Аллергии:</b> {', '.join(user_data.get('allergies', [])) or 'Не указаны'}", styles['Normal']))
    story.append(Spacer(1, 24))
    
    # Рекомендации
    rec_title = Paragraph("<b>Ваш персональный план ухода:</b>", styles['Heading2'])
    story.append(rec_title)
    story.append(Spacer(1, 12))
    
    daily_routine = recommendations.get('daily_routine', [])
    if daily_routine:
        for idx, item in enumerate(daily_routine, 1):
            symptoms_text = ", ".join(user_data.get('symptoms', [])) if user_data.get('symptoms') else "проблем не обнаружено"
            user_text = f"""
            У тебя {user_data.get('skin_type', 'не определён').lower()} кожа.
            {"Мы заметили у тебя небольшую особенность — " + symptoms_text.lower() + "." 
             if user_data.get('symptoms') 
             else "Проблем не обнаружено! Вы молодец, что так замечательно ухаживаете за кожей лица!"}
            Рекомендуем попробовать {item.get('method', 'Уходовая косметика')} 
            {item.get('type', 'Крем')} в течение {item.get('course_duration', 30)} дней.
            """
            # Заголовок рекомендации
            rec_header = Paragraph(f"Рекомендация {idx}: {item.get('method')} - {item.get('type')}", styles['RecommendationTitle'])
            story.append(rec_header)
            story.append(Paragraph(user_text, styles['Normal']))
            
            # Очищаем инструкцию от HTML и Markdown
            instruction = item.get('template', 'Следуйте рекомендациям специалиста')
            instruction = instruction.replace('<div class="recommendation-text">', '').replace('</div>', '')
            instruction = instruction.replace('<p>', '').replace('</p>', '<br/>')
            instruction = instruction.replace('**', '').replace('*', '')
            
            # Обрабатываем противопоказания
            contraindications = item.get('contraindications', ['Не указаны'])
            if isinstance(contraindications, list):
                # Если это список, преобразуем его в строку
                contraindications = ', '.join(contraindications)
            # Удаляем лишние запятые и пробелы
            contraindications = contraindications.replace(',,', ',').strip()
            # Заменяем запятые на тег <br/> для корректного переноса в PDF
            contraindications = contraindications.replace(', ', ',<br/>')

            details = [
                f"Инструкция:<br/>{instruction}",
                f"Эффективность: {item.get('effectiveness', 'Средняя')}%",
                f"Противопоказания: {contraindications}",
                f"Ожидаемые результаты: {item.get('expected_results', 'Не указаны')}"
            ]
            for detail in details:
                story.append(Paragraph(detail, styles['Bullet']))
            story.append(Spacer(1, 16))  # Увеличиваем отступ между рекомендациями
    else:
        story.append(Paragraph("Рекомендаций пока нет.", styles['Normal']))
    
    story.append(Spacer(1, 24))
    footnote_text = "*Эти советы не заменяют консультацию косметолога. Перед применением новых средств рекомендуется провести тест на аллергическую реакцию."
    footnote = Paragraph(footnote_text, footnote_style)
    story.append(footnote)
    
    # Подпись
    story.append(Spacer(1, 12))
    signature = Paragraph("© 2025 Beauty Tracker | Создано с заботой для вашей кожи 🌸", footnote_style)
    story.append(signature)
    
    doc.build(story)
    return filename