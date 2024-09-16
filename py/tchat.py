from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_community.llms import GPT4All
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage

# Инициализация хранилища для сессий:
store = {}

# Функция для получения истории сессии:
# Эта функция проверяет, существует ли история для заданного session_id.
# Если нет, она создает новую историю в памяти.
# Затем возвращает объект истории сообщений для данной сессии.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Создание системного сообщения и шаблона чата:
# system_template: Определяет системное сообщение, которое инструктирует модель отвечать на русском языке и, при необходимости, делать обоснованные предположения.
# messages: Список, содержащий системное сообщение и плейсхолдер для сообщений пользователя.
# prompt: Шаблон чата, объединяющий системное сообщение и сообщения пользователя.
system_template = """If you don't know the answer, make up your best guess. You must answer in Russian language."""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="messages"),
]

# Создаем промпт
prompt = ChatPromptTemplate(messages=messages)

# Инициализация модели GPT4All, с 8 потоками вычислений:
model = GPT4All(model="/Users/andreykruglik/Downloads/Llama-2-13B-Storywriter-LORA/mistral.7b.smart-lemon-cookie.gguf_v2.q4_k_m.gguf", n_threads=8)

# Цепь (Chain) соединяющая промпт и модель
# Оператор | используется для объединения шаблона чата и модели в одну цепочку, где вывод шаблона передается на вход модели.
chain = prompt | model

# Оборачиваем цепь с управлением историей сообщений
# Эта обертка позволяет сохранять и использовать историю сообщений в ходе сессии чата.
with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

# Main loop for interaction
while True:
    content = input(">> ")

    # Определяем session_id для переписки. Может быть статическим или динамическим.
    session_id = "default_session"

    # Инпут должен содержать ключ 'messages'
    input_data = {
        "messages": [
            # HumanMessagePromptTemplate.from_template(content).format(content=content)
            HumanMessage(content=content)
        ],
    }

    # Конфиг содержащий session_id
    config = {"configurable": {"session_id": session_id}}

    # Метод invoke запускает цепочку обработки с учетом истории сообщений.
    result = with_message_history.invoke(input=input_data, config=config)
    print(result)

    # Как работает:
    # Ввод пользователя: Программа ждет ввода сообщения от пользователя.
    # Определение session_id: Используется идентификатор сессии для отслеживания истории (в данном случае статический).
    # Подготовка входных данных: Создается словарь input_data с ключом messages, содержащим отформатированное сообщение пользователя.
    # Конфигурация сессии: Параметр config передает session_id для управления историей сообщений.
    # Вызов модели: Метод invoke запускает цепочку обработки с учетом истории сообщений.
    # Вывод результата: Ответ модели выводится на экран.
