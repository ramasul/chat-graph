#Import built-in libraries
import os
import gc
import time
import json
import base64
from typing import List
from datetime import datetime, timezone

#Import third-party libraries
import uvicorn
import asyncio
import secrets

#Import FastAPI libraries
from fastapi_health import health
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form

from Secweb.XContentTypeOptions import XContentTypeOptions
from Secweb.XFrameOptions import XFrame

from langchain_neo4j import Neo4jGraph

#Import Starlette libraries
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import HTMLResponse

#Import Local Libraries
from src.graphQA import *
from src.shared.utils import *
from src.api_response import create_api_response
from src.logger import CustomLogger
# from src.chat_interaction import *

logger = CustomLogger()
CHUNK_DIR = os.path.join(os.path.dirname(__file__), "chunks")
MERGED_DIR = os.path.join(os.path.dirname(__file__), "merged_files")

def healthy_condition():
    output = {"healthy": True}
    return output

def healthy():
    return True

def sick():
    return False

class CustomGZipMiddleware:
    """[ENG]: Custom GZip Middleware to compress responses.
    [IDN]: Middleware GZip untuk mengompres respon."""

    def __init__(
        self,
        app: ASGIApp,
        paths: List[str],
        minimum_size: int = 1000,
        compresslevel: int = 5
    ):
        self.app = app
        self.paths = paths
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
 
        path = scope["path"]
        should_compress = any(path.startswith(gzip_path) for gzip_path in self.paths)
        
        if not should_compress:
            return await self.app(scope, receive, send)
        
        gzip_middleware = GZipMiddleware(
            app=self.app,
            minimum_size=self.minimum_size,
            compresslevel=self.compresslevel
        )
        await gzip_middleware(scope, receive, send)

app = FastAPI()
app.add_middleware(XContentTypeOptions)
app.add_middleware(XFrame, Option={'X-Frame-Options': 'DENY'})
app.add_middleware(CustomGZipMiddleware, minimum_size=1000, compresslevel=5,paths=["/sources_list","/url/scan","/extract","/chat_bot","/chunk_entities","/get_neighbours","/graph_query","/schema","/populate_graph_schema","/get_unconnected_nodes_list","/get_duplicate_nodes","/fetch_chunktext"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))
app.add_api_route("/health", health([healthy_condition, healthy]))

# Serve static files (including favicon.ico)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Treatment Recommender System API</title>
            <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="flex items-center justify-center h-screen bg-gray-100">
            <div class="bg-white shadow-md rounded-lg p-8 text-center">
                <h1 class="text-2xl font-bold text-gray-800">Welcome to the Treatment Recommender System Backend</h1>
                <p class="mt-4 text-gray-600">Click the button below to see the API documentation:</p>
                <a href="/docs">
                    <button class="mt-4 px-6 py-2 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 transition duration-200">
                        View API Docs
                    </button>
                </a>
            </div>
        </body>
    </html>
    """

@app.post("/chat_bot/diagnose")
async def chat_bot(uri=Form(),model=Form(None),userName=Form(), password=Form(), database=Form(),question=Form(None), document_names=Form(None),session_id=Form(None),mode=Form(None),email=Form()):
    logging.info(f"QA_RAG called at {datetime.now()}")
    qa_rag_start_time = time.time()
    try:
        if mode == "graph":
            graph = Neo4jGraph( url=uri,username=userName,password=password,database=database,sanitize = True, refresh_schema=True)
        else:
            graph = create_graph_database_connection(uri, userName, password, database)
        
        result = await asyncio.to_thread(QA_RAG,graph=graph,model=model,question=question,document_names=document_names,session_id=session_id,mode=mode)

        total_call_time = time.time() - qa_rag_start_time
        logging.info(f"Total Response time is  {total_call_time:.2f} seconds")
        result["info"]["response_time"] = round(total_call_time, 2)
        
        json_obj = {'api_name':'chat_bot','db_url':uri, 'userName':userName, 'database':database, 'question':question,'document_names':document_names,
                             'session_id':session_id, 'mode':mode, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{total_call_time:.2f}','email':email}
        logger.log_struct(json_obj, "INFO")
        
        return create_api_response('Success',data=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to get chat response"
        error_message = str(e)
        logging.exception(f'Exception in chat bot:{error_message}')
        return create_api_response(job_status, message=message, error=error_message,data=mode)
    finally:
        gc.collect()


@app.post("/clear_chat_bot")
async def clear_chat_bot(uri=Form(),userName=Form(), password=Form(), database=Form(), session_id=Form(None),email=Form()):
    try:
        start = time.time()
        graph = create_graph_database_connection(uri, userName, password, database)
        result = await asyncio.to_thread(clear_chat_history,graph=graph,session_id=session_id)
        end = time.time()
        elapsed_time = end - start
        json_obj = {'api_name':'clear_chat_bot', 'db_url':uri, 'userName':userName, 'database':database, 'session_id':session_id, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{elapsed_time:.2f}','email':email}
        logger.log_struct(json_obj, "INFO")
        return create_api_response('Success',data=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to clear chat History"
        error_message = str(e)
        logging.exception(f'Exception in chat bot:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()
     
@app.post("/chat_bot/interact")
async def chat_bot(
    model: str = Form(),
    human_messages: str = Form(),
    session_id: str = Form(),
    context: Optional[str] = Form(None),
    diagnosis: bool = Form(False),
    disease_context: Optional[str] = Form(None)
):
    logging.info(f"Chat interaction called at {datetime.now()}")
    start_time = time.time()
    try:
        context_dict = json.loads(context) if context else None
        result = await asyncio.to_thread(
            chat_interaction,
            model=model,
            human_messages=human_messages,
            session_id=session_id,
            context=context_dict,
            diagnosis=diagnosis,
            disease_context=disease_context
        )
        total_call_time = time.time() - start_time
        result["info"]["response_time"] = round(total_call_time, 2)
        return create_api_response('Success',data=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to clear chat History"
        error_message = str(e)
        logging.exception(f'Exception in chat bot:{error_message}')
        return create_api_response(job_status, message=message, error=error_message)
    finally:
        gc.collect()


@app.post("/check-symptoms")
def check_symptoms(human_messages=Form(), model= Form(), session_id=Form()):
    result = check_if_chat_is_symptoms(human_messages, model, session_id)
    return {"is_symptoms": result}


@app.post("/init_chat")
async def initialize_chat(session_id=Form(None), context=Form(None)):
    logging.info(f"Initialize_chat called at {datetime.now()}")
    init_chat_start_time = time.time()
    try:
        context = json.loads(context) if context else None
        result = await asyncio.to_thread(initial_greeting,session_id=session_id,context=context)

        total_call_time = time.time() - init_chat_start_time
        logging.info(f"Total Response time is  {total_call_time:.2f} seconds")
        result["info"]["response_time"] = round(total_call_time, 2)
        
        json_obj = {'api_name':'init_chat','session_id':session_id, 'Context':context, 'logging_time': formatted_time(datetime.now(timezone.utc)), 'elapsed_api_time':f'{total_call_time:.2f}'}
        logger.log_struct(json_obj, "INFO")
        
        return create_api_response('Success',data=result)
    except Exception as e:
        job_status = "Failed"
        message="Unable to get chat response"
        error_message = str(e)
        logging.exception(f'Exception in chat bot:{error_message}')
        return create_api_response(job_status, message=message, error=error_message,data=session_id)
    finally:
        gc.collect()

if __name__ == "__main__":    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)