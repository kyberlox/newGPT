from fastapi import FastAPI, Body, Response, Cookie
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from typing import Annotated, List

import openai
from openai import OpenAI
from openai import AsyncOpenAI



import json
import os
from dotenv import load_dotenv

import asyncio

import base64

import uuid

import tempfile


load_dotenv()
key = os.getenv('key')
organization = os.getenv('organization')
project_id = os.getenv('project_id')



client = OpenAI(api_key = key)
async_client = AsyncOpenAI(api_key = key)



app = FastAPI()

link = "127.0.0.1:8000"
web = "https://gpt.emk.ru/"

origins = [
    '*',
    web,
    "https://178.217.101.144/",
    "https://gpt.emk.ru",
    "https://portal.emk.ru",
    "http://intranet.emk.org.ru",
    "http://intranet.emk.ru"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"]
    #allow_headers=["Content-Type", "Accept", "Location", "Allow", "Content-Disposition", "Sec-Fetch-Dest"],
)

'''try:
        lg = output.choices[0].message.content
        log = f"\n \n :=> \n {params} => \n {lg}"
        l = open('log.txt', 'r')
        old = l.read()
        l.close()
        l = open('log.txt', 'w')
        l.write(old + log)
        l.close()

        if os.path.getsize("log.txt") > 1073741824:
            l = open('log.txt', 'w')
            l.close()
    except:
        print("ошибка в теле запроса: ", params)
'''

@app.post("/")
def root(data = Body()):
    print(data)
    params = json.loads(data.decode('UTF-8'))
    #params = data
    print(params)
    model = "gpt-4o-mini"
    if "model" in params:
        model = params["model"]
    else:
        return {"error" : "The system has recorded your IP address.  There will be a check."}
    output = client.chat.completions.create(
        model = model,
        messages = params["messages"],
        temperature = params["temperature"],
        max_tokens = params["max_tokens"]
        )

    return output

@app.post("/api")
def api(data = Body()):
    params = data
    print(params)
    output = client.chat.completions.create(
        model = params["model"],
        messages = params["messages"],
        temperature = params["temperature"],
        max_tokens = params["max_tokens"]
        ).choices[0].message.content

    return output

#диалоговое общение
@app.post("/dialog")
async def dialog(data=Body()):
    print(data)
    #если задана модель
    model = "gpt-4o-mini"
    if "model" in data:
        model = data["model"]

    #читаем диалог
    messages = []
    #диалог с параметрами
    if "messages" in data:
        messages = data["messages"]
    #переписка
    else:
        messages = data

    response = await async_client.chat.completions.create(
        model = model,
        messages = messages
    )

    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    return messages



'''
#анализ файла
@app.post("/send_file")
def upload_file(file: UploadFile, data=Body()):
    #если задана модель
    model = "gpt-4o-mini"
    if "model" in data:
        model = data["model"]

    #читаем диалог
    messages = []
    #диалог с параметрами
    if "messages" in data:
        messages = data["messages"]
    #переписка
    else:
        messages = data

    #просто декодируем в base64
    #with open("path/to/image.png", "rb") as image_file:
        #b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    #и дописываем в messages
    #file_promt{"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"}
    file_promt{"type": file.content_type, "image_url": f"data:{file.content_type};base64,{b64_image}"}

    return file
'''

# Разрешенные типы изображений
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

@app.post("/analyze-image")
async def create_upload_files(files: List[UploadFile], data : dict = Body()): #, prompt: str = "Что изображено на картинках?"):
    if "prompt" in data:
        promt = data["promt"]
    else:
        prompt = "Что изображено на картинках?"
    try:
        files_urls = []
        # Обработка каждого файла
        for file in files:

            # Проверяем, что файл - изображение
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Неподдерживаемый тип файла. Разрешены: {', '.join(ALLOWED_IMAGE_TYPES.keys())}"
                )

            # Проверка размера (макс. 10MB)
            max_size = 10 * 1024 * 1024
            file.file.seek(0, 2)
            file_size = file.file.tell()
            if file_size > max_size:
                raise HTTPException(status_code=413, detail="Изображение слишком большое (максимум 10MB)")
            file.file.seek(0)

            # Читаем файл и кодируем в base64
            image_bytes = await file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            file_url = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file.content_type};base64,{base64_image}"
                },
            }

            files_urls.append(file_url)


        content = files_urls
        content.append({"type": "text", "text": prompt})

        # Отправляем в OpenAI GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1000,
        )

        # Получаем ответ
        analysis = response.choices[0].message.content

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "image_type": file.content_type,
            "analysis": analysis,
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

openai.api_key = key

@app.post("/generate-image")
async def generate_image(data=Body()):  
    """
    Генерирует изображение через DALL·E 3 (возвращает только URL)
    
    Параметры:
    - prompt: описание изображения
    - size: размер (1024x1024, 1024x1792 или 1792x1024)
    - quality: качество ("standard" или "hd")
    - style: стиль ("vivid" или "natural")
    """

    prompt: str
    if "prompt" in data:
        prompt = data["prompt"]
    else:
        raise HTTPException(status_code=500, detail="Invilid token!")

    size = "1024x1024"
    if "size" in data:
        size = data["size"]

    quality = "standard"
    if "quality" in data:
        quality = data["quality"]

    style = "vivid"
    if "style" in data:
        style = data["style"]

    try:
        # Вызов DALL·E 3
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1
        )

        return JSONResponse({
            "status": "success",
            "image_url": response.data[0].url,  # URL изображения (живет 2 часа)
            "revised_prompt": response.data[0].revised_prompt  # Оптимизированный запрос
        })

    except openai.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"Неверный запрос: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")



@app.post("/analyze-files")
async def analyze_files(files: List[UploadFile], data : dict = Body()):
    """Анализирует различные типы файлов"""
    print(data)
    if "prompt" in data.keys:
        prompt = data["prompt"]
    else:
        prompt = "Проанализируй содержимое этих файлов"
    
    try:
        files_content = []
        
        for file in files:
            file_bytes = await file.read()
            
            if file.content_type.startswith("image/"):
                # Для изображений используем base64
                base64_image = base64.b64encode(file_bytes).decode("utf-8")
                files_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file.content_type};base64,{base64_image}"
                    }
                })
            else:
                # Для документов используем текстовое извлечение
                extracted_text = await extract_text_from_document(file_bytes, file.content_type, file.filename)
                files_content.append({
                    "type": "text", 
                    "text": f"Содержимое файла '{file.filename}':\n\n{extracted_text}"
                })
        
        # Добавляем промпт
        files_content.append({"type": "text", "text": prompt})
        
        # Отправляем в OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": files_content
                }
            ],
            max_tokens=2000
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "success": True,
            "files_processed": [file.filename for file in files],
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")



async def extract_text_from_document(file_bytes: bytes, content_type: str, filename: str) -> str:
    """Извлекает текст из документов простым способом"""
    try:
        if content_type == "application/pdf":
            # Для PDF используем простой текстовый парсер
            import PyPDF2
            import io
            pdf_file = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_file.pages:
                text += page.extract_text() + "\n"
            return text if text.strip() else "Не удалось извлечь текст из PDF файла"
        
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Для DOCX используем простой парсер
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text if text.strip() else "Не удалось извлечь текст из Word документа"
        
        elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                             "application/vnd.ms-excel"]:
            # Для Excel используем простой парсер
            import pandas as pd
            import io
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            text = f"Файл Excel содержит {len(excel_file.sheet_names)} листов: {', '.join(excel_file.sheet_names)}\n\n"
            
            for sheet_name in excel_file.sheet_names[:2]:  # Ограничиваем первыми 2 листами
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=10)  # Ограничиваем 10 строками
                text += f"Лист: {sheet_name} (строк: {len(df)}, столбцов: {len(df.columns)})\n"
                text += "Первые строки данных:\n"
                text += df.head(3).to_string() + "\n\n"  # Только первые 3 строки
            return text
        
        elif content_type in ["text/plain", "text/csv"]:
            # Для текстовых файлов
            try:
                return file_bytes.decode('utf-8')[:5000]  # Ограничиваем размер
            except:
                return file_bytes.decode('latin-1')[:5000]
        
        else:
            return f"Формат файла {content_type} не поддерживается для автоматического извлечения текста"
            
    except Exception as e:
        return f"Ошибка при извлечении текста: {str(e)}"

async def process_documents_with_assistant(document_files: list, prompt: str) -> str:
    """Обрабатывает документы через Assistants API"""
    try:
        # Загружаем файлы в OpenAI
        file_ids = []
        for doc_file in document_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=doc_file["filename"]) as temp_file:
                temp_file.write(doc_file["bytes"])
                temp_path = temp_file.name
            
            try:
                with open(temp_path, "rb") as file_stream:
                    uploaded_file = client.files.create(
                        file=file_stream,
                        purpose="assistants"
                    )
                file_ids.append(uploaded_file.id)
            finally:
                os.unlink(temp_path)
        
        # Создаем векторное хранилище
        vector_store = client.vector_stores.create(
            name="Document Analysis"
        )
        
        # Добавляем файлы в векторное хранилище
        for file_id in file_ids:
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file_id
            )
        
        # Создаем ассистента
        assistant = client.beta.assistants.create(
            instructions="Ты - помощник для анализа документов. Анализируй содержимое файлов и отвечай на вопросы.",
            model="gpt-4-turbo",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store.id]
                }
            }
        )
        
        # Создаем тред и запускаем
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "file_ids": file_ids
                }
            ]
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Ждем завершения
        while run.status not in ["completed", "failed", "cancelled", "expired"]:
            await asyncio.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        
        if run.status == "completed":
            # Получаем ответ
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            analysis = messages.data[0].content[0].text.value
        else:
            analysis = f"Ошибка обработки документов: {run.status}"
        
        # Очистка ресурсов
        try:
            client.beta.assistants.delete(assistant.id)
            for file_id in file_ids:
                client.files.delete(file_id)
            client.vector_stores.delete(vector_store.id)
        except:
            pass  # Игнорируем ошибки очистки
        
        return analysis
        
    except Exception as e:
        return f"Ошибка при обработке документов: {str(e)}"

async def process_images_with_chat(image_files: list, prompt: str) -> str:
    """Обрабатывает изображения через Chat Completions"""
    try:
        content = [{"type": "text", "text": prompt}]
        
        for image_file in image_files:
            base64_image = base64.b64encode(image_file["bytes"]).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_file['content_type']};base64,{base64_image}"
                }
            })
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Ошибка при обработке изображений: {str(e)}"