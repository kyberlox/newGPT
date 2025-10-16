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



@app.post("/analyze-files")
async def analyze_files(files: List[UploadFile], data = Body()):
    """Анализирует различные типы файлов"""
    
    if "prompt" in data:
        prompt = data["prompt"]
    else:
        prompt = "Проанализируй содержимое этих файлов"
    
    try:
        # Загружаем файлы
        file_ids = []
        for file in files:
            file_bytes = await file.read()
            
            temp_filename = f"temp_{file.filename}"
            with open(temp_filename, "wb") as f:
                f.write(file_bytes)
            
            try:
                with open(temp_filename, "rb") as file_stream:
                    uploaded_file = client.files.create(
                        file=file_stream,
                        purpose="assistants"
                    )
                file_ids.append(uploaded_file.id)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        
        # Создаем ассистента
        assistant = client.beta.assistants.create(
            instructions="Ты - помощник для анализа файлов.",
            model="gpt-4-turbo",
            tools=[{"type": "retrieval"}]
        )
        
        # Создаем тред и добавляем сообщение с файлами
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "file_ids": file_ids
                }
            ]
        )
        
        # Запускаем ассистента
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Ждем завершения
        import time
        while run.status not in ["completed", "failed"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
        
        if run.status == "completed":
            # Получаем сообщения
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            analysis = messages.data[0].content[0].text.value
            
            # Очищаем
            client.beta.assistants.delete(assistant.id)
            for file_id in file_ids:
                client.files.delete(file_id)
            
            return {
                "success": True,
                "files_processed": [file.filename for file in files],
                "analysis": analysis
            }
        else:
            raise HTTPException(status_code=500, detail=f"Ошибка выполнения: {run.status}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")