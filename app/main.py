from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import shutil
import uuid
import json
import logging
from datetime import datetime
from typing import Dict

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.utils.llm_client import LLMClient
from app.services.comment_analyzer import CommentAnalyzer
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = DocumentProcessor()
vector_store = VectorStoreService()
llm_client = LLMClient()
comment_analyzer = CommentAnalyzer(vector_store, llm_client)

app = FastAPI(title="Document Analysis API")

results_store: Dict[str, Dict] = {}

os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Анализ Документов</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="file"] {
                display: block;
                margin-bottom: 10px;
                width: 100%;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            .info {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #17a2b8;
                margin: 20px 0;
            }
            .steps {
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 4px;
                margin: 20px 0;
            }
            .steps ol {
                margin-top: 10px;
                padding-left: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Сервис Анализа Документов</h1>
        <div class="info">
            <p>Этот сервис анализирует две версии документа и определяет, были ли учтены комментарии из первой версии во второй версии.</p>
        </div>
        <div class="steps">
            <h3>Как это работает:</h3>
            <ol>
                <li>Загрузите первую версию вашего документа (PDF или TXT)</li>
                <li>Загрузите вторую версию с изменениями (PDF или TXT)</li>
                <li>Загрузите файл CSV с комментариями (первый столбец должен содержать комментарии)</li>
                <li>Система проанализирует, был ли каждый комментарий учтён в обновлённом документе</li>
                <li>Просмотрите подробный отчет анализа со статусом для каждого комментария</li>
            </ol>
        </div>
        <form action="/analyze/" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="doc_v1">Документ Версия 1:</label>
                <input type="file" name="doc_v1" accept=".pdf,.txt" required>
            </div>
            <div class="form-group">
                <label for="doc_v2">Документ Версия 2:</label>
                <input type="file" name="doc_v2" accept=".pdf,.txt" required>
            </div>
            <div class="form-group">
                <label for="comments">Комментарии (CSV):</label>
                <input type="file" name="comments" accept=".csv" required>
            </div>
            <button type="submit">Анализировать документы</button>
        </form>
    </body>
    </html>
    """


@app.post("/analyze/")
async def analyze_documents(
    background_tasks: BackgroundTasks,
    doc_v1: UploadFile = File(...),
    doc_v2: UploadFile = File(...),
    comments: UploadFile = File(...),
):
    task_id = f"task_{uuid.uuid4().hex}"

    task_dir = os.path.join("uploads", task_id)
    os.makedirs(task_dir, exist_ok=True)

    doc_v1_path = os.path.join(task_dir, f"v1_{doc_v1.filename}")
    doc_v2_path = os.path.join(task_dir, f"v2_{doc_v2.filename}")
    comments_path = os.path.join(task_dir, comments.filename)

    for file_path, upload_file in [
        (doc_v1_path, doc_v1),
        (doc_v2_path, doc_v2),
        (comments_path, comments),
    ]:
        try:
            with open(file_path, "wb") as f:
                content = await upload_file.read()
                f.write(content)
        except Exception as e:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file {upload_file.filename}: {e}",
            )

    doc_base_name = os.path.splitext(os.path.basename(doc_v1.filename))[0]
    results_store[task_id] = {
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "doc_v1": doc_v1.filename,
        "doc_v2": doc_v2.filename,
        "comments": comments.filename,
        "base_name": doc_base_name,
        "progress": "uploading",
        "result": None,
        "results_path": None,
    }

    background_tasks.add_task(
        process_documents,
        task_id,
        doc_v1_path,
        doc_v2_path,
        comments_path,
        doc_base_name,
    )

    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Обработка документов</title>
            <meta http-equiv="refresh" content="2;url=/status/{task_id}/html?t={int(datetime.now().timestamp())}">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                }}
                .loader {{
                    border: 8px solid #f3f3f3;
                    border-top: 8px solid #3498db;
                    border-radius: 50%;
                    width: 60px;
                    height: 60px;
                    animation: spin 2s linear infinite;
                    margin: 30px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                .info {{
                    background-color: #e9f7fe;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 20px 0;
                    text-align: left;
                }}
            </style>
        </head>
        <body>
            <h1>Ваши документы обрабатываются</h1>
            <div class="loader"></div>
            <p>Пожалуйста, подождите пока мы анализируем ваши документы. Вы будете перенаправлены автоматически.</p>
            <div class="info">
                <p><strong>Документ V1:</strong> {doc_v1.filename}</p>
                <p><strong>Документ V2:</strong> {doc_v2.filename}</p>
                <p><strong>Комментарии:</strong> {comments.filename}</p>
            </div>
        </body>
        </html>
        """
    )


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")

    result = results_store[task_id]

    if result["status"] == "completed" and result.get("results_path"):
        result["results_url"] = f"/results/{os.path.basename(result['results_path'])}"

    return result


@app.get("/status/{task_id}/html", response_class=HTMLResponse)
async def get_status_html(task_id: str):
    if task_id not in results_store:
        return HTMLResponse(
            """
            <html><body>
            <h1>Задача не найдена</h1>
            <p>Запрашиваемая задача не существует или истек срок ее хранения.</p>
            <p><a href="/">Вернуться на главную</a></p>
            </body></html>
            """,
            status_code=404,
        )

    result = results_store[task_id]
    status = result["status"]

    progress_mapping = {
        "uploading": {"text": "Загрузка документов", "percent": 10},
        "processing_v1": {"text": "Обработка документа версии 1", "percent": 20},
        "processing_v2": {"text": "Обработка документа версии 2", "percent": 40},
        "processing_comments": {"text": "Обработка комментариев", "percent": 60},
        "creating_vector_db": {"text": "Создание векторной базы данных", "percent": 70},
        "analyzing_comments": {
            "text": "Анализ комментариев с помощью ИИ",
            "percent": 80,
        },
        "saving_results": {"text": "Сохранение результатов анализа", "percent": 95},
    }

    if status == "processing":
        progress = result.get("progress", "unknown")
        progress_text = "Processing documents"
        progress_percent = 0

        if progress in progress_mapping:
            progress_info = progress_mapping[progress]
            progress_text = progress_info["text"]
            progress_percent = progress_info["percent"]

        if progress == "analyzing_comments":
            comment_progress = result.get("comment_progress", "")
            if comment_progress:
                progress_text += f" ({comment_progress})"

        html = f"""
        <html>
        <head>
            <title>Обработка документов</title>
            <meta http-equiv="refresh" content="3;url=/status/{task_id}/html?t={int(datetime.now().timestamp())}">
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                .processing {{ background-color: #fff3cd; padding: 15px; border-radius: 4px; }}
                .progress-container {{
                    margin: 20px 0;
                    height: 24px;
                    background-color: #f3f3f3;
                    border-radius: 12px;
                    overflow: hidden;
                }}
                .progress-bar {{
                    height: 24px;
                    width: {progress_percent}%;
                    background-color: #4CAF50;
                    text-align: center;
                    line-height: 24px;
                    color: white;
                    transition: width 0.5s;
                }}
                .step-list {{ margin-top: 20px; }}
                .step {{ padding: 8px; margin: 5px 0; }}
                .step.active {{
                    background-color: #e7f3ff;
                    border-left: 4px solid #0275d8;
                    font-weight: bold;
                }}
                .step.completed {{
                    background-color: #e8f5e9;
                    border-left: 4px solid #4CAF50;
                    color: #388e3c;
                }}
            </style>
        </head>
        <body>
            <h1>Обработка документов</h1>
            <div class="processing">
                <p><strong>Текущий статус:</strong> {progress_text}</p>
                <div class="progress-container">
                    <div class="progress-bar">{progress_percent}%</div>
                </div>
                <div class="step-list">
                    <div class="step {'completed' if progress_percent > 10 else 'active' if progress == 'uploading' else ''}">
                        Загрузка документов
                    </div>
                    <div class="step {'completed' if progress_percent > 20 else 'active' if progress == 'processing_v1' else ''}">
                        Обработка документа версии 1: {result['doc_v1']}
                    </div>
                    <div class="step {'completed' if progress_percent > 40 else 'active' if progress == 'processing_v2' else ''}">
                        Обработка документа версии 2: {result['doc_v2']}
                    </div>
                    <div class="step {'completed' if progress_percent > 60 else 'active' if progress == 'processing_comments' else ''}">
                        Обработка комментариев: {result['comments']}
                    </div>
                    <div class="step {'completed' if progress_percent > 70 else 'active' if progress == 'creating_vector_db' else ''}">
                        Создание векторной базы данных
                    </div>
                    <div class="step {'completed' if progress_percent > 80 else 'active' if progress == 'analyzing_comments' else ''}">
                        Анализ комментариев с помощью ИИ {result.get('comment_progress', '')}
                    </div>
                    <div class="step {'completed' if progress_percent > 95 else 'active' if progress == 'saving_results' else ''}">
                        Сохранение результатов анализа
                    </div>
                </div>
                <p style="margin-top: 20px;">Эта страница будет автоматически обновляться каждые 3 секунды.</p>
            </div>
        </body>
        </html>
        """

    elif status == "completed":
        html = f"""
        <html>
        <head>
            <title>Анализ завершён</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .success {{ background-color: #d4edda; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
                .comment-container {{ border: 1px solid #ddd; margin-bottom: 20px; border-radius: 8px; overflow: hidden; }}
                .comment-header {{ background-color: #f8f8f8; padding: 10px 15px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }}
                .comment-header h3 {{ margin: 0; }}
                .comment-body {{ padding: 15px; }}
                .comment-text {{ font-weight: bold; margin-bottom: 10px; }}
                .status {{ padding: 5px 10px; border-radius: 4px; display: inline-block; margin: 5px 0; }}
                .status-учтен {{ background-color: #d4edda; color: #155724; }}
                .status-частично-учтен {{ background-color: #fff3cd; color: #856404; }}
                .status-не-учтен {{ background-color: #f8d7da; color: #721c24; }}
                .status-error {{ background-color: #f8d7da; color: #721c24; }}
                .evidence {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #6c757d; white-space: pre-wrap; font-family: monospace; }}
                .explanation {{ margin-bottom: 15px; white-space: pre-wrap; }}
                .suggestion {{ background-color: #e2f0fb; padding: 10px; margin: 10px 0; border-left: 4px solid #0275d8; white-space: pre-wrap; }}
                .btn {{ background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block; margin-right: 10px; }}
                .summary {{ margin-bottom: 25px; }}
                .summary-stats {{ display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap; }}
                .summary-stat {{ padding: 15px; border-radius: 8px; flex: 1; text-align: center; min-width: 150px; }}
                .stat-addressed {{ background-color: #d4edda; }}
                .stat-partially {{ background-color: #fff3cd; }}
                .stat-not {{ background-color: #f8d7da; }}
                .toggle-btn {{ background-color: #6c757d; color: white; padding: 5px 10px; cursor: pointer; border: none; border-radius: 4px; }}
            </style>
            <script>
                function toggleDetails(commentId) {{
                    var details = document.getElementById('details-' + commentId);
                    var button = document.getElementById('btn-' + commentId);
                    if (details.style.display === 'none' || details.style.display === '') {{
                        details.style.display = 'block';
                        button.textContent = 'Скрыть детали';
                    }} else {{
                        details.style.display = 'none';
                        button.textContent = 'Показать детали';
                    }}
                }}
            </script>
        </head>
        <body>
            <h1>Результаты анализа документа</h1>
            <div class="success">
                <p>Анализ документа успешно завершен.</p>
                <p><b>Документ V1:</b> {result.get('doc_v1', 'Н/Д')}</p>
                <p><b>Документ V2:</b> {result.get('doc_v2', 'Н/Д')}</p>
                <p><b>Комментарии:</b> {result.get('comments', 'Н/Д')}</p>
                {'<a href="' + f"/results/{os.path.basename(result['results_path'])}" + '" class="btn" download>Скачать полный анализ (JSON)</a>' if result.get('results_path') else ''}
                <a href="/" class="btn" style="background-color: #6c757d;">Вернуться на главную</a>
            </div>
            <div class="summary">
                <h2>Сводка анализа</h2>
        """

        results_data = result.get("result", [])
        total = len(results_data)
        addressed = sum(1 for c in results_data if c.get("status") == "учтен")
        partially = sum(1 for c in results_data if c.get("status") == "частично учтен")
        not_addressed = sum(1 for c in results_data if c.get("status") == "не учтен")
        error_count = sum(1 for c in results_data if c.get("status") == "error")

        html += f"""
                <div class="summary-stats">
                    <div class="summary-stat stat-addressed">
                        <h3>{addressed}</h3>
                        <p>Учтены</p>
                    </div>
                    <div class="summary-stat stat-partially">
                        <h3>{partially}</h3>
                        <p>Частично учтены</p>
                    </div>
                    <div class="summary-stat stat-not">
                        <h3>{not_addressed}</h3>
                        <p>Не учтены</p>
                    </div>
                    {f'<div class="summary-stat status-error"><h3>{error_count}</h3><p>Ошибка</p></div>' if error_count > 0 else ''}
                </div>
            </div>

            <h2>Детальный анализ ({total} комментариев)</h2>
        """

        for i, comment in enumerate(results_data):
            comment_id = comment.get("comment_id", i + 1)
            comment_text = comment.get("comment_text", "Н/Д")
            status = comment.get("status", "error")
            explanation = comment.get("explanation", "No explanation available.")
            evidence_v1 = comment.get("evidence_v1", "No evidence found in V1.")
            evidence_v2 = comment.get("evidence_v2", "No evidence found in V2.")
            suggestion = comment.get("suggestion", "")

            status_class = f"status-{status.replace(' ', '-')}"

            html += f"""
            <div class="comment-container">
                <div class="comment-header">
                    <h3>Комментарий {comment_id}</h3>
                    <button id="btn-{i}" onclick="toggleDetails('{i}')" class="toggle-btn">Показать детали</button>
                </div>
                <div class="comment-body">
                    <div class="comment-text">{comment_text}</div>
                    <div class="status {status_class}">{status}</div>
                    <div id="details-{i}" style="display: none;">
                        <h4>Объяснение:</h4>
                        <div class="explanation">{explanation}</div>
                        <h4>Подтверждение из V1:</h4>
                        <div class="evidence">{evidence_v1}</div>
                        <h4>Подтверждение из V2:</h4>
                        <div class="evidence">{evidence_v2}</div>
                        {f'<h4>Предложение:</h4><div class="suggestion">{suggestion}</div>' if suggestion else ''}
                    </div>
                </div>
            </div>
            """

        html += """
        </body>
        </html>
        """
    else:
        error_message = result.get("error", "Unknown error occurred")
        html = f"""
        <html>
        <head>
            <title>Ошибка обработки</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                .error {{ background-color: #f8d7da; padding: 15px; border-radius: 4px; }}
                .btn {{ background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Ошибка обработки</h1>
            <div class="error">
                <p>Произошла ошибка во время анализа документа:</p>
                <p><b>{error_message}</b></p>
                <p>Документы: <b>{result.get('doc_v1', 'Н/Д')}</b> и <b>{result.get('doc_v2', 'Н/Д')}</b></p>
                <p>Комментарии: <b>{result.get('comments', 'Н/Д')}</b></p>
            </div>
            <a href="/" class="btn">Вернуться на главную</a>
        </body>
        </html>
        """

    return HTMLResponse(content=html, status_code=200)


async def process_documents(
    task_id: str,
    doc_v1_path: str,
    doc_v2_path: str,
    comments_path: str,
    doc_base_name: str,
):
    try:
        logger.info(f"Starting processing for task {task_id}")

        results_store[task_id]["progress"] = "processing_v1"
        await asyncio.sleep(0)

        processed_v1 = processor.process_document(doc_v1_path)
        if not processed_v1 or not processed_v1.get("chunks"):
            raise Exception(
                f"Failed to process document v1 or no text extracted: {doc_v1_path}"
            )

        results_store[task_id]["progress"] = "processing_v2"
        await asyncio.sleep(0)

        processed_v2 = processor.process_document(doc_v2_path)
        if not processed_v2 or not processed_v2.get("chunks"):
            raise Exception(
                f"Failed to process document v2 or no text extracted: {doc_v2_path}"
            )

        results_store[task_id]["progress"] = "processing_comments"
        await asyncio.sleep(0)

        processed_comments = processor.process_comments(comments_path)
        if not processed_comments:
            raise Exception(
                f"Failed to process comments or no comments found: {comments_path}"
            )

        results_store[task_id]["progress"] = "creating_vector_db"
        await asyncio.sleep(0)

        clean_base_name = doc_base_name
        for prefix in [
            settings.QDRANT_COLLECTION_V1_PREFIX,
            settings.QDRANT_COLLECTION_V2_PREFIX,
        ]:
            if clean_base_name.startswith(prefix):
                clean_base_name = clean_base_name[len(prefix) :]

        v1_collection = (
            f"{settings.QDRANT_COLLECTION_V1_PREFIX}{clean_base_name}-{task_id}"
        )
        v2_collection = (
            f"{settings.QDRANT_COLLECTION_V2_PREFIX}{clean_base_name}-{task_id}"
        )

        vector_store.recreate_collection(v1_collection)
        vector_store.upsert_chunks(v1_collection, processed_v1["chunks"])

        vector_store.recreate_collection(v2_collection)
        vector_store.upsert_chunks(v2_collection, processed_v2["chunks"])

        results_store[task_id]["progress"] = "analyzing_comments"
        await asyncio.sleep(0)

        results = []
        total_comments = len(processed_comments)

        analysis_base_name = f"{clean_base_name}-{task_id}"

        for i, comment in enumerate(processed_comments):
            results_store[task_id]["comment_progress"] = f"{i+1}/{total_comments}"
            await asyncio.sleep(0)

            logger.info(f"Task {task_id}: Analyzing comment {i+1}/{total_comments}")

            result = comment_analyzer.analyze_comment(comment, analysis_base_name)
            results.append(result)

        results_store[task_id]["progress"] = "saving_results"
        await asyncio.sleep(0)

        results_file = f"results_{task_id}.json"
        results_path = os.path.join("results", results_file)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        results_store[task_id].update(
            {
                "status": "completed",
                "progress": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": results,
                "results_path": results_path,
            }
        )

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        results_store[task_id].update(
            {
                "status": "error",
                "progress": "error",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            }
        )


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Document Analysis API")

    logger.info(f"vLLM URL: http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1")
    logger.info(f"Qdrant URL: http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    for directory in ["results", "uploads"]:
        os.makedirs(directory, exist_ok=True)
        try:
            os.chmod(directory, 0o777)
        except Exception as e:
            logger.warning(f"Failed to set permissions on {directory}: {e}")

    results_dir = "results"
    loaded_count = 0

    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith("results_") and filename.endswith(".json"):
                try:
                    task_id = filename[8:-5]
                    filepath = os.path.join(results_dir, filename)

                    with open(filepath, "r", encoding="utf-8") as f:
                        results_data = json.load(f)

                    results_store[task_id] = {
                        "status": "completed",
                        "progress": "completed",
                        "created_at": datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).isoformat(),
                        "completed_at": datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat(),
                        "result": results_data,
                        "results_path": filepath,
                    }
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load result file {filename}: {e}")

    logger.info(f"Loaded {loaded_count} existing results from disk")

    cleanup_old_files(days=7)


def cleanup_old_files(days=7):
    now = datetime.now()
    results_dir = "results"

    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith("results_") and filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if (now - mtime).days > days:
                        task_id = filename[8:-5]
                        os.remove(filepath)
                        if task_id in results_store:
                            del results_store[task_id]
                        logger.info(f"Cleaned up old result file: {filename}")
                except Exception as e:
                    logger.warning(
                        f"Failed to process file during cleanup: {filepath}, {e}"
                    )
