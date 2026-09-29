import os
import re
import chromadb
from chromadb.utils import embedding_functions

DOCS_FOLDER = "documents"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "snip_docs"


def find_context(text, chunk_start):
    pattern = r'(\d+(?:\.\d+)*)\s+[А-Яа-я]'
    matches = list(re.finditer(pattern, text[:chunk_start]))
    if matches:
        return matches[-1].group(1)
    return ""


def find_tables(text):
    tables = re.findall(r'Таблиц[аы]\s+([А-ЯA-Z]?\.?\d+(?:\.\d+)?)', text)
    return tables


def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk, start))
        start = end - overlap
    return chunks


def build_database(progress_callback=None):
    client = chromadb.PersistentClient(path=DB_PATH)

    ru_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="cointegrated/rubert-tiny2"
    )

    try:
        client.delete_collection(name=COLLECTION_NAME)
        if progress_callback:
            progress_callback("Старая база удалена. Создаю новую...")
    except Exception:
        if progress_callback:
            progress_callback("Создаю новую базу...")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ru_ef
    )

    if not os.path.exists(DOCS_FOLDER):
        raise FileNotFoundError(
            f"Папка '{DOCS_FOLDER}' не найдена! "
            f"Убедись, что документы загружены в репозиторий."
        )

    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"В папке '{DOCS_FOLDER}' нет .txt файлов!")

    if progress_callback:
        progress_callback(f"Найдено документов: {len(files)}")

    for filename in files:
        filepath = os.path.join(DOCS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_text(content)
        if progress_callback:
            progress_callback(f"  {filename}: разбит на {len(chunks)} кусочков")

        for i, (chunk, start_pos) in enumerate(chunks):
            section = find_context(content, start_pos)
            tables = find_tables(chunk)
            table_info = ", ".join(tables) if tables else ""

            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "chunk_id": i,
                    "section": section,
                    "tables": table_info
                }],
                ids=[f"{filename}_{i}"]
            )

    if progress_callback:
        progress_callback("✅ База данных создана!")

    return collection