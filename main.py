from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import uvicorn

app = FastAPI()

# Подключение шаблонов и статики
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Глобальная переменная для хранения данных
tfidf_data = []


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {
        "request": request, "data": None, "page": 1, "pages": 1
    })


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Обработка файла и вычисление TF-IDF"""
    global tfidf_data

    content = await file.read()
    text = content.decode("utf-8")

    # Разделение текста на слова
    words = pd.Series(text.split())

    # TF-IDF вычисление
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

    # Формирование данных для отображения
    df = pd.DataFrame(list(tfidf_scores.items()), columns=["word", "tfidf"])
    df["tf"] = df["word"].apply(lambda x: text.split().count(x))
    df = df.sort_values("tfidf", ascending=False).reset_index(drop=True)

    tfidf_data = df.to_dict(orient="records")

    return templates.TemplateResponse("index.html", {
        "request": request, "data": tfidf_data[:10], "page": 1, "pages": (len(tfidf_data) // 10) + 1
    })


@app.get("/page/{page}")
async def get_page(request: Request, page: int = 1, per_page: int = Query(10)):
    """Вывод данных постранично"""
    global tfidf_data

    if not tfidf_data:
        return templates.TemplateResponse("index.html", {
            "request": request, "data": None, "page": 1, "pages": 1
        })

    # Пагинация
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = (len(tfidf_data) // per_page) + (1 if len(tfidf_data) % per_page > 0 else 0)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "data": tfidf_data[start:end],
        "page": page,
        "pages": total_pages
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
