from fastapi import FastAPI, Body, Response, Cookie
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from openai import OpenAI
from openai import AsyncOpenAI



import json
import os
from dotenv import load_dotenv

import asyncio

import base64



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
    web,
    "https://178.217.101.144/",
    "https://gpt.emk.ru",
    "https://portal.emk.ru"
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
ALLOWED_MIME_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile, data=Body()):
    # Проверяем, что файл - изображение
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла. Разрешены: {', '.join(ALLOWED_MIME_TYPES.keys())}"
        )

    # Проверка размера (макс. 10MB)
    max_size = 10 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    if file_size > max_size:
        raise HTTPException(status_code=413, detail="Изображение слишком большое (максимум 10MB)")
    file.file.seek(0)

    try:
        # Читаем файл и кодируем в base64
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Отправляем в OpenAI GPT-4o
        '''
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{base64_image}"
                            },
                        },
                    ],
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
        '''

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

        current_file_response = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file.content_type};base64,{base64_image}"
                    },
                },
            ],
        }

        messages.append(current_file_response)

        return messages


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")