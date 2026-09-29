import streamlit as st
import chromadb
from openai import OpenAI
import re

# Настройка страницы
st.set_page_config(
    page_title="Поиск по СНиПам",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Поиск по строительным нормам")
st.markdown("Задайте вопрос — программа найдёт ответ в СП и СНиПах.")

# Подключаемся к модели Qwen (ключ берётся из секретов Streamlit)
@st.cache_resource
def load_client():
    api_key = st.secrets["CLOUD_API_KEY"]
    return OpenAI(
        api_key=api_key,
        base_url="https://foundation-models.api.cloud.ru/v1"
    )

client = load_client()

# Подключаемся к векторной базе (кэшируем)
@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_collection(name="snip_docs")

collection = load_collection()

# Боковая панель с информацией
with st.sidebar:
    st.header("📚 База знаний")
    st.write("Загружено документов: **5**")
    st.markdown("---")
    st.markdown("**Документы:**")
    st.write("• ГОСТ Р 51872-2024")
    st.write("• СП 126.13330.2017")
    st.write("• СП 46.13330.2012")
    st.write("• СП 70.13330.2012")
    st.write("• СП 78.13330.2012")

# Поле для вопроса
question = st.text_input("Ваш вопрос:", placeholder="Например: допуски по асфальту")

# Кнопка поиска
if st.button("🔍 Найти ответ", type="primary"):
    if not question.strip():
        st.warning("Пожалуйста, введите вопрос.")
    else:
        with st.spinner("Ищу ответ в документах..."):
            candidates = []
            
            # 1. ВЕКТОРНЫЙ ПОИСК (30 фрагментов)
            vector_results = collection.query(
                query_texts=[question],
                n_results=30
            )
            if vector_results['documents'] and vector_results['documents'][0]:
                for i, doc in enumerate(vector_results['documents'][0]):
                    meta = vector_results['metadatas'][0][i]
                    candidates.append({
                        'text': doc,
                        'source': meta['source'],
                        'section': meta.get('section', ''),
                        'tables': meta.get('tables', ''),
                        'type': 'векторный'
                    })
            
            # 2. ПОИСК ПО ТАБЛИЦАМ (если в вопросе есть "таблица")
            table_matches = re.findall(r'таблиц[аы]?\s*([А-ЯA-Z]?\.?\d+(?:\.\d+)?)', question, re.IGNORECASE)
            if table_matches:
                for table_num in table_matches:
                    st.info(f"🔍 Найдено упоминание таблицы {table_num}. Ищу точное совпадение...")
                    try:
                        table_results = collection.get(
                            where_document={"$contains": f"Таблица {table_num}"},
                            limit=10
                        )
                        if table_results['documents']:
                            for i, doc in enumerate(table_results['documents']):
                                meta = table_results['metadatas'][i]
                                candidates.append({
                                    'text': doc,
                                    'source': meta['source'],
                                    'section': meta.get('section', ''),
                                    'tables': meta.get('tables', ''),
                                    'type': f'таблица {table_num}'
                                })
                    except Exception as e:
                        st.warning(f"Ошибка поиска таблицы: {e}")
            
            # 3. ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ (допуск, отклонение)
            if re.search(r'допуск|отклонени', question, re.IGNORECASE):
                keyword_results = collection.query(
                    query_texts=["допуск отклонение не более мм"],
                    n_results=20
                )
                if keyword_results['documents'] and keyword_results['documents'][0]:
                    for i, doc in enumerate(keyword_results['documents'][0]):
                        meta = keyword_results['metadatas'][0][i]
                        candidates.append({
                            'text': doc,
                            'source': meta['source'],
                            'section': meta.get('section', ''),
                            'tables': meta.get('tables', ''),
                            'type': 'ключевые слова'
                        })
            
            # 4. ФИЛЬТРАЦИЯ
            filtered = []
            for c in candidates:
                text_lower = c['text'].lower()
                has_number = bool(re.search(r'\d+', c['text']))
                has_keyword = any(word in text_lower for word in ['допуск', 'отклонен', 'мм', 'таблиц', 'не более'])
                if has_number and has_keyword:
                    filtered.append(c)
            
            if not filtered:
                filtered = candidates
            
            # Убираем дубликаты
            seen = set()
            unique_filtered = []
            for c in filtered:
                if c['text'] not in seen:
                    seen.add(c['text'])
                    unique_filtered.append(c)
            
            unique_filtered = unique_filtered[:25]
            
            # Собираем контекст и источники
            context = ""
            sources = []
            for c in unique_filtered:
                ref = c['source']
                if c['section']:
                    ref += f", раздел {c['section']}"
                if c['tables']:
                    ref += f", таблица {c['tables']}"
                context += f"\n\n--- Источник: {ref} ---\n{c['text']}"
                if ref not in sources:
                    sources.append(ref)
            
            st.info(f"📖 Отобрано фрагментов: {len(unique_filtered)}")
            st.markdown("**Источники:**")
            for src in sources[:10]:
                st.write(f"• {src}")
            
            # Формируем запрос к модели
            prompt = f"""Не размышляй. Сразу давай ответ.
Ты — эксперт по строительным нормам и правилам.
Отвечай подробно. Приведи ВСЕ найденные допуски и отклонения из фрагментов.
Структурируй ответ: раздели на пункты, для каждого укажи значение и источник.
Если в фрагментах нет ответа — честно скажи об этом.
Обязательно укажи, из какого документа, раздела и пункта взята информация.

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ВОПРОС:
{question}
"""
            
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen3.6-35B-A3B",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=6000
                )
                
                answer = response.choices[0].message.content
                
                st.success("Ответ найден!")
                st.markdown("### 📖 Ответ:")
                st.markdown(answer)
                
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")