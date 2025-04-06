def list_to_text(x):
    """Преобразует список строк в строку."""
    if isinstance(x, list):
        return ' '.join(x)
    return str(x)

def custom_tokenizer(x):
    """Разбивает строку на слова по пробелам."""
    return x.split()