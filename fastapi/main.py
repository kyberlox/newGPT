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

import pandas as pd
import json
import xml.etree.ElementTree as ET
from docx import Document
import PyPDF2
import io


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
async def create_upload_files(files: List[UploadFile], data=Body()): #, prompt: str = "Что изображено на картинках?"):
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



# Расширенные разрешенные типы файлов
ALLOWED_MIME_TYPES = {
    # Изображения
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    
    # Документы
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/plain": "txt",
    "text/csv": "csv",
}

def extract_text_from_file(file_bytes: bytes, content_type: str, filename: str) -> str:
    """Извлекает текст из файла в зависимости от его типа"""
    
    try:
        if content_type == "application/pdf":
            return extract_text_from_pdf(file_bytes)
        
        elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            return extract_text_from_docx(file_bytes)
        
        elif content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            return extract_text_from_excel(file_bytes)
        
        elif content_type in ["application/json", "text/json"]:
            return extract_text_from_json(file_bytes)
        
        elif content_type in ["application/xml", "text/xml"]:
            return extract_text_from_xml(file_bytes)
        
        elif content_type in ["text/plain", "text/csv"]:
            return file_bytes.decode('utf-8')
        
        else:
            return f"Формат файла {content_type} не поддерживается для текстового анализа"
            
    except Exception as e:
        return f"Ошибка при извлечении текста из файла {filename}: {str(e)}"

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Извлекает текст из PDF"""
    pdf_file = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_file.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Извлекает текст из Word документа"""
    doc = Document(io.BytesIO(file_bytes))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_excel(file_bytes: bytes) -> str:
    """Извлекает текст из Excel файла"""
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    text = ""
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        text += f"Лист: {sheet_name}\n"
        text += df.to_string() + "\n\n"
    
    return text

def extract_text_from_json(file_bytes: bytes) -> str:
    """Извлекает текст из JSON файла"""
    data = json.loads(file_bytes.decode('utf-8'))
    return json.dumps(data, ensure_ascii=False, indent=2)

def extract_text_from_xml(file_bytes: bytes) -> str:
    """Извлекает текст из XML файла"""
    root = ET.fromstring(file_bytes.decode('utf-8'))
    return ET.tostring(root, encoding='unicode', method='xml')

@app.post("/analyze-files")
async def analyze_files(files: List[UploadFile], data = Body()):
    """Анализирует различные типы файлов"""
    
    if "prompt" in data:
        prompt = data["prompt"]
    else:
        prompt = "Проанализируй содержимое этих файлов"
    
    try:
        uploaded_file_ids = []
        
        # Загрузка файлов на серверы OpenAI
        for file in files:
            # Читаем содержимое файла
            content = await file.read()
            
            # Создаем временный файл
            temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
            with open(temp_filename, "wb") as f:
                f.write(content)
            
            # Загружаем в OpenAI
            try:
                with open(temp_filename, "rb") as file_stream:
                    uploaded_file = client.files.create(
                        file=file_stream,
                        purpose="assistants"
                    )
                uploaded_file_ids.append(uploaded_file.id)
            finally:
                # Удаляем временный файл
                os.remove(temp_filename)
        
        # Создаем векторное хранилище для поиска по файлам
        vector_store = client.beta.vector_stores.create(
            name="Analysis Files"
        )
        
        # Добавляем файлы в векторное хранилище
        for file_id in uploaded_file_ids:
            client.beta.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file_id
            )
        
        # Создаем ассистента с доступом к файлам
        assistant = client.beta.assistants.create(
            instructions="Вы - помощник для анализа документов и изображений.",
            model="gpt-4-turbo",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store.id]
                }
            }
        )
        
        # Создаем тред и отправляем сообщение
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "attachments": [
                        {
                            "file_id": file_id,
                            "tools": [{"type": "file_search"}]
                        }
                        for file_id in uploaded_file_ids
                    ]
                }
            ]
        )
        
        # Запускаем ассистента
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Ожидаем завершения
        while run.status in ["queued", "in_progress"]:
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
            
            return {
                "success": True,
                "files_processed": [file.filename for file in files],
                "analysis": analysis
            }
        else:
            raise HTTPException(status_code=500, detail=f"Ошибка анализа: {run.status}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


    """Анализирует только документы (без изображений)"""
    
    if "prompt" in data:
        prompt = data["prompt"]
    else:
        prompt = "Проанализируй содержимое этих документов"
    
    try:
        combined_text = ""
        
        for file in files:
            # Проверяем, что это не изображение
            if file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="Этот endpoint предназначен только для документов. Используйте /analyze-files для изображений."
                )

            if file.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Неподдерживаемый тип файла: {file.content_type}"
                )

            # Проверка размера
            max_size = 10 * 1024 * 1024
            file.file.seek(0, 2)
            file_size = file.file.tell()
            if file_size > max_size:
                raise HTTPException(status_code=413, detail=f"Файл {file.filename} слишком большой")
            file.file.seek(0)

            # Извлекаем текст
            file_bytes = await file.read()
            extracted_text = extract_text_from_file(file_bytes, file.content_type, file.filename)
            combined_text += f"\n\n--- Файл: {file.filename} ---\n{extracted_text}"

        # Отправляем в OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + combined_text}
                    ]
                }
            ],
            max_tokens=3000,
        )

        analysis = response.choices[0].message.content

        return JSONResponse({
            "success": True,
            "files_processed": [file.filename for file in files],
            "analysis": analysis,
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")