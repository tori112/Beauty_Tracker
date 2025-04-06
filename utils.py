def list_to_text(x):
    """Преобразует pandas.Series со списками строк в pandas.Series со строками."""
    return x.apply(lambda lst: ' '.join(lst))

def custom_tokenizer(x):
    """Разбивает строку на слова по пробелам."""
    return x.split()