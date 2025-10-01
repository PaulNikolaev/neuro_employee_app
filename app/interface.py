import json
import re
import gradio as gr
from .gpt_model import GPT

# Загружаем данные из JSON
with open("app/data.json", "r", encoding="utf-8") as f:
    models = json.load(f)

# Создаем экземпляр класса GPT с выбранной моделью LLM
gpt = GPT("gpt-3.5-turbo")


def create_interface():
    with gr.Blocks() as demo:
        with gr.Tab("Выбор модели и Промт"):
            # Dropdown выбора модели
            subject = gr.Dropdown(
                choices=[(m["name"], i) for i, m in enumerate(models)],
                label="Выбор модели",
                value=0  # Начальное значение
            )

            # Поля для отображения данных модели
            name = gr.Label(value=models[0]['name'], show_label=False)
            prompt = gr.Textbox(
                value=re.sub(r'\t+|\s\s+', ' ', models[0]['prompt']),
                label="Промт",
                interactive=True,
                lines=3
            )
            query = gr.Textbox(value=models[0]['query'], label="Запрос к LLM", interactive=True)
            link = gr.HTML(value=f"<a target='_blank' href='{models[0]['doc']}'>Документ для обучения</a>")

            # Функция изменения выбора модели
            def onchange(dropdown_index):
                try:
                    index = int(dropdown_index)
                    model = models[index]
                    return [
                        gr.update(value=model['name']),
                        gr.update(value=re.sub(r'\t+|\s\s+', ' ', model['prompt'])),
                        gr.update(value=model['query']),
                        gr.update(value=f"<a target='_blank' href='{model['doc']}'>Документ для обучения</a>")
                    ]
                except Exception as e:
                    return [
                        gr.update(value="Ошибка"),
                        gr.update(value=""),
                        gr.update(value=""),
                        gr.update(value=f"Ошибка: {str(e)}")
                    ]

            subject.change(onchange, inputs=[subject], outputs=[name, prompt, query, link])

            # Кнопки действий
            with gr.Row():
                train_btn = gr.Button("Обучить модель")
                request_btn = gr.Button("Запрос к модели")

            # Поля вывода результатов
            with gr.Row():
                response = gr.Textbox(label="Ответ LLM", interactive=False, lines=10)
                log = gr.Textbox(label="Логирование", interactive=False, lines=10)

            # Функция обучения
            def train(dropdown_index):
                try:
                    index = int(dropdown_index)
                    gpt.load_search_indexes(models[index]['doc'])
                    return gpt.log
                except Exception as e:
                    return f"Ошибка при обучении: {str(e)}"

            # Функция запроса к модели
            def predict(prompt_text, query_text):
                try:
                    result = gpt.answer_index(prompt_text, query_text)

                    # Выводим только важные строки логов
                    log_lines = [line for line in gpt.log.splitlines() if "Токенов" in line or "Ошибка" in line]
                    filtered_log = "\n".join(log_lines)

                    return [result, filtered_log]
                except Exception as e:
                    return ["", f"Ошибка: {str(e)}"]

            # Привязка функций к кнопкам
            train_btn.click(train, [subject], [log])
            request_btn.click(predict, [prompt, query], [response, log])

    return demo