import streamlit.components.v1 as components

def apply_custom_styles():
    components.html(
        """
        <script>
        function updateDropdownStyles() {
            // Находим все элементы selectbox и multiselect
            const selectboxes = document.querySelectorAll('[data-baseweb="select"]');
            selectboxes.forEach(selectbox => {
                // Основной контейнер
                selectbox.style.backgroundColor = '#FFFFFF';
                selectbox.style.color = '#333333';
                selectbox.style.border = '1px solid #FF9999';
                selectbox.style.borderRadius = '10px';

                // Внутренние элементы
                const innerDivs = selectbox.querySelectorAll('div');
                innerDivs.forEach(div => {
                    div.style.backgroundColor = '#FFFFFF';
                    div.style.color = '#333333';
                });

                // Текст внутри
                const spans = selectbox.querySelectorAll('span');
                spans.forEach(span => {
                    span.style.color = '#333333';
                });

                // Выпадающий список (ul)
                const uls = selectbox.querySelectorAll('ul');
                uls.forEach(ul => {
                    ul.style.backgroundColor = '#FFFFFF';
                    ul.style.border = '1px solid #FF9999';
                });

                // Элементы списка (li)
                const lis = selectbox.querySelectorAll('li');
                lis.forEach(li => {
                    li.style.backgroundColor = '#FFFFFF';
                    li.style.color = '#333333';
                });

                // Эффект наведения
                lis.forEach(li => {
                    li.addEventListener('mouseover', () => {
                        li.style.backgroundColor = '#FF9999';
                        li.style.color = '#FFFFFF';
                    });
                    li.addEventListener('mouseout', () => {
                        li.style.backgroundColor = '#FFFFFF';
                        li.style.color = '#333333';
                    });
                });
            });
        }

        // Выполняем функцию после загрузки страницы
        window.onload = updateDropdownStyles;

        // Наблюдаем за изменениями в DOM
        const observer = new MutationObserver((mutations) => {
            updateDropdownStyles();
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Периодически проверяем (на случай, если observer не сработает)
        setInterval(updateDropdownStyles, 500);
        </script>
        """,
        height=0  # Компонент не должен занимать место на странице
    )