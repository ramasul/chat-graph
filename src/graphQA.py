import os
import json
import time
import logging

import threading
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from langchain_neo4j import Neo4jVector
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_text_splitters import TokenTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.callbacks import BaseCallbackHandler

# LangChain chat models
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Local imports
from src.llm import get_llm
from src.shared.utils import load_embedding_model
from src.shared.constants import *
import json
import logging
from datetime import datetime
from src.llm import get_llm
load_dotenv() 

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
EMBEDDING_FUNCTION , _ = load_embedding_model(EMBEDDING_MODEL) 

class SessionChatHistory:
    history_dict = {}

    @classmethod
    def get_chat_history(cls, session_id):
        """Retrieve or create chat message history for a given session ID."""
        if session_id not in cls.history_dict:
            logging.info(f"Creating new ChatMessageHistory Local for session ID: {session_id}")
            cls.history_dict[session_id] = ChatMessageHistory()
        else:
            logging.info(f"Retrieved existing ChatMessageHistory Local for session ID: {session_id}")
        return cls.history_dict[session_id]
    
class CustomCallback(BaseCallbackHandler):

    def __init__(self):
        self.transformed_question = None
    
    def on_llm_end(
        self,response, **kwargs: Any
    ) -> None:
        logging.info("question transformed")
        self.transformed_question = response.generations[0][0].text.strip()

def get_history_by_session_id(session_id):
    try:
        return SessionChatHistory.get_chat_history(session_id)
    except Exception as e:
        logging.error(f"Failed to get history for session ID '{session_id}': {e}")
        raise 

def get_total_tokens(ai_response, llm):
    try:
        if isinstance(llm, (ChatOpenAI, ChatGroq)):
            total_tokens = ai_response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
        
        else:
            logging.warning(f"Unrecognized language model: {type(llm)}. Returning 0 tokens.")
            total_tokens = 0

    except Exception as e:
        logging.error(f"Error retrieving total tokens: {e}")
        total_tokens = 0

    return total_tokens

def clear_chat_history(session_id, local=True, graph=None):
    try:
        if not local:
            history = Neo4jChatMessageHistory(
                graph=graph,
                session_id=session_id
            )
        else:
            history = get_history_by_session_id(session_id)
        
        history.clear()

        return {
            "session_id": session_id, 
            "message": "The chat history has been cleared.", 
            "user": "chatbot"
        }
    
    except Exception as e:
        logging.error(f"Error clearing chat history for session {session_id}: {e}")
        return {
            "session_id": session_id, 
            "message": "Failed to clear chat history.", 
            "user": "chatbot"
        }

def get_sources_and_chunks(sources_used, docs):
    chunkdetails_list = []
    sources_used_set = set(sources_used)
    seen_ids_and_scores = set()  

    for doc in docs:
        try:
            source = doc.metadata.get("source")
            chunkdetails = doc.metadata.get("chunkdetails", [])

            if source in sources_used_set:
                for chunkdetail in chunkdetails:
                    id = chunkdetail.get("id")
                    score = round(chunkdetail.get("score", 0), 4)

                    id_and_score = (id, score)

                    if id_and_score not in seen_ids_and_scores:
                        seen_ids_and_scores.add(id_and_score)
                        chunkdetails_list.append({**chunkdetail, "score": score})

        except Exception as e:
            logging.error(f"Error processing document: {e}")

    result = {
        'sources': sources_used,
        'chunkdetails': chunkdetails_list,
    }
    return result

def get_rag_chain(llm, system_template = CHAT_SYSTEM_TEMPLATE):
    try:
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name = "messages"),
                (
                    "human",
                    "User question: {input}"
                ),
            ]
        )

        question_answering_chain = question_answering_prompt | llm

        return question_answering_chain

    except Exception as e:
        logging.error(f"Error creating RAG chain: {e}")
        raise

def format_documents(documents, model):
    prompt_token_cutoff = 4
    for model_names, value in CHAT_TOKEN_CUT_OFF.items():
        if model in model_names:
            prompt_token_cutoff = value
            break

    sorted_documents = sorted(documents, key = lambda doc: doc.state.get("query_similarity_score", 0), reverse = True)
    sorted_documents = sorted_documents[:prompt_token_cutoff]

    formatted_docs = list()
    sources = set()
    entities = dict()

    for doc in sorted_documents:
        try:
            source = doc.metadata.get('source', "unknown")
            sources.add(source)

            entities = doc.metadata['entities'] if 'entities'in doc.metadata.keys() else entities
            formatted_doc = (
                "Document start\n"
                f"This Document belongs to the source {source}\n"
                f"Content: {doc.page_content}\n"
                "Document end\n"
            )
            formatted_docs.append(formatted_doc)
        
        except Exception as e:
            logging.error(f"Error formatting document: {e}")
    
    return "\n\n".join(formatted_docs), sources, entities

def process_documents(docs, question, messages, llm, model, chat_mode_settings):
    start_time = time.time() 
    try:
        formatted_docs, sources, entitydetails = format_documents(docs, model)
        
        rag_chain = get_rag_chain(llm = llm)
        
        ai_response = rag_chain.invoke({
            "messages": messages[:-1],
            "context": formatted_docs,
            "input": question
        })

        result = {'sources': list(), 'nodedetails': dict(), 'entities': dict()}
        node_details = {"chunkdetails":list(),"entitydetails":list(),"communitydetails":list()}
        entities = {'entityids':list(),"relationshipids":list()}

        sources_and_chunks = get_sources_and_chunks(sources, docs)
        result['sources'] = sources_and_chunks['sources']
        node_details["chunkdetails"] = sources_and_chunks["chunkdetails"]
        entities.update(entitydetails)    

        result["nodedetails"] = node_details
        result["entities"] = entities

        content = ai_response.content
        total_tokens = get_total_tokens(ai_response, llm)
        
        predict_time = time.time() - start_time
        logging.info(f"Final response predicted in {predict_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        raise
    
    return content, result, total_tokens, formatted_docs

def retrieve_documents(doc_retriever, messages):
    start_time = time.time()

    try:
        handler = CustomCallback()
        docs = doc_retriever.invoke({"messages": messages},{"callbacks":[handler]})
        transformed_question = handler.transformed_question
        if transformed_question:
            logging.info(f"Transformed question : {transformed_question}")
        doc_retrieval_time = time.time() - start_time
        logging.info(f"Documents retrieved in {doc_retrieval_time:.2f} seconds")
        
    except Exception as e:
        error_message = f"Error retrieving documents: {str(e)}"
        logging.error(error_message)
        docs = None
        transformed_question = None
    
    return docs, transformed_question

def create_document_retriever_chain(llm, retriever):
    """[ENG]: Create a document retriever chain that transforms the user's question before passing it to the retriever.
    [IDN]: Buat chain pengambil dokumen yang mentransformasi pertanyaan pengguna sebelum meneruskannya ke pengambil."""
    try:
        logging.info("Starting to create document retriever chain")

        query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUESTION_TRANSFORM_TEMPLATE),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        output_parser = StrOutputParser()

        splitter = TokenTextSplitter(chunk_size=CHAT_DOC_SPLIT_SIZE, chunk_overlap=0)
        embeddings_filter = EmbeddingsFilter(
            embeddings=EMBEDDING_FUNCTION,
            similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD
        )

        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, embeddings_filter]
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )

        query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                (lambda x: x["messages"][-1].content) | compression_retriever,
            ),
            query_transform_prompt | llm | output_parser | compression_retriever,
        ).with_config(run_name="chat_retriever_chain")

        logging.info("Successfully created document retriever chain")
        return query_transforming_retriever_chain

    except Exception as e:
        logging.error(f"Error creating document retriever chain: {e}", exc_info=True)
        raise

def initialize_neo4j_vector(graph, chat_mode_settings):
    try:
        retrieval_query = chat_mode_settings.get("retrieval_query")
        index_name = chat_mode_settings.get("index_name")
        keyword_index = chat_mode_settings.get("keyword_index", "")
        node_label = chat_mode_settings.get("node_label")
        embedding_node_property = chat_mode_settings.get("embedding_node_property")
        text_node_properties = chat_mode_settings.get("text_node_properties")


        if not retrieval_query or not index_name:
            raise ValueError("Required settings 'retrieval_query' or 'index_name' are missing.")

        if keyword_index:
            neo_db = Neo4jVector.from_existing_graph(
                embedding=EMBEDDING_FUNCTION,
                index_name=index_name,
                retrieval_query=retrieval_query,
                graph=graph,
                search_type="hybrid",
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties,
                keyword_index_name=keyword_index
            )
            logging.info(f"Successfully retrieved Neo4jVector Fulltext index '{index_name}' and keyword index '{keyword_index}'")
        else:
            neo_db = Neo4jVector.from_existing_graph(
                embedding=EMBEDDING_FUNCTION,
                index_name=index_name,
                retrieval_query=retrieval_query,
                graph=graph,
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties
            )
            logging.info(f"Successfully retrieved Neo4jVector index '{index_name}'")
    except Exception as e:
        index_name = chat_mode_settings.get("index_name")
        logging.error(f"Error retrieving Neo4jVector index {index_name} : {e}")
        raise
    return neo_db

def create_retriever(neo_db, document_names, chat_mode_settings, search_k, score_threshold, ef_ratio):
    if document_names and chat_mode_settings["document_filter"]:
        retriever = neo_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'k': search_k,
                'effective_search_ratio': ef_ratio,
                'score_threshold': score_threshold,
                'filter': {'fileName': {'$in': document_names}}
            }
        )
        logging.info(f"Successfully created retriever with search_k={search_k}, score_threshold={score_threshold} for documents {document_names}")
    else:
        retriever = neo_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': search_k,'effective_search_ratio': ef_ratio, 'score_threshold': score_threshold}
        )
        logging.info(f"Successfully created retriever with search_k={search_k}, score_threshold={score_threshold}")
    return retriever

def get_neo4j_retriever(graph, document_names,chat_mode_settings, score_threshold=CHAT_SEARCH_KWARG_SCORE_THRESHOLD):
    try:

        neo_db = initialize_neo4j_vector(graph, chat_mode_settings)
        search_k = chat_mode_settings["top_k"]
        ef_ratio = int(os.getenv("EFFECTIVE_SEARCH_RATIO", "2")) if os.getenv("EFFECTIVE_SEARCH_RATIO", "2").isdigit() else 2
        retriever = create_retriever(neo_db, document_names,chat_mode_settings, search_k, score_threshold,ef_ratio)
        return retriever
    except Exception as e:
        index_name = chat_mode_settings.get("index_name")
        logging.error(f"Error retrieving Neo4jVector index  {index_name} or creating retriever: {e}")
        raise Exception(f"An error occurred while retrieving the Neo4jVector index or creating the retriever. Please drop and create a new vector index '{index_name}': {e}") from e 

def setup_chat(model, graph, document_names, chat_mode_settings):
    start_time = time.time()
    try:
        if model == "diffbot":
            model = os.getenv('DEFAULT_DIFFBOT_CHAT_MODEL')
        
        llm, model_name = get_llm(model=model)
        logging.info(f"Model called in chat: {model} (version: {model_name})")

        retriever = get_neo4j_retriever(graph=graph, chat_mode_settings=chat_mode_settings, document_names=document_names)
        doc_retriever = create_document_retriever_chain(llm, retriever)
        
        chat_setup_time = time.time() - start_time
        logging.info(f"Chat setup completed in {chat_setup_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during chat setup: {e}", exc_info=True)
        raise
    
    return llm, doc_retriever, model_name

def process_chat_response(messages, history, question, model, graph, document_names, chat_mode_settings):
    try:
        llm, doc_retriever, model_version = setup_chat(model, graph, document_names, chat_mode_settings)
        
        docs, transformed_question = retrieve_documents(doc_retriever, messages)  

        if docs:
            content, result, total_tokens,formatted_docs = process_documents(docs, question, messages, llm, model, chat_mode_settings)
        else:
            content = "I couldn't find any relevant documents to answer your question."
            result = {"sources": list(), "nodedetails": list(), "entities": list()}
            total_tokens = 0
            formatted_docs = ""
        
        ai_response = AIMessage(content=content)
        messages.append(ai_response)

        summarization_thread = threading.Thread(target=summarize_and_log, args=(history, messages, llm))
        summarization_thread.start()
        logging.info("Summarization thread started.")
        # summarize_and_log(history, messages, llm)
        metric_details = {"question":question,"contexts":formatted_docs,"answer":content}
        return {
            "session_id": "",  
            "message": content,
            "info": {
                # "metrics" : metrics,
                "sources": result["sources"],
                "model": model_version,
                "nodedetails": result["nodedetails"],
                "total_tokens": total_tokens,
                "response_time": 0,
                "mode": chat_mode_settings["mode"],
                "entities": result["entities"],
                "metric_details": metric_details,
            },
            
            "user": "chatbot"
        }
    
    except Exception as e:
        logging.exception(f"Error processing chat response at {datetime.now()}: {str(e)}")
        return {
            "session_id": "",
            "message": "Something went wrong",
            "info": {
                "metrics" : [],
                "sources": [],
                "nodedetails": [],
                "total_tokens": 0,
                "response_time": 0,
                "error": f"{type(e).__name__}: {str(e)}",
                "mode": chat_mode_settings["mode"],
                "entities": [],
                "metric_details": {},
            },
            "user": "chatbot"
        }

def summarize_and_log(history, stored_messages, llm):
    logging.info("Starting summarization in a separate thread.")
    if not stored_messages:
        logging.info("No messages to summarize.")
        return False

    try:
        start_time = time.time()

        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    "Summarize the above chat messages into a concise message, focusing on key points and relevant details that could be useful for future conversations. Exclude all introductions and extraneous information."
                ),
            ]
        )
        summarization_chain = summarization_prompt | llm

        summary_message = summarization_chain.invoke({"chat_history": stored_messages})

        with threading.Lock():
            history.clear()
            history.add_user_message("Our current conversation summary till now")
            history.add_message(summary_message)

        history_summarized_time = time.time() - start_time
        logging.info(f"Chat History summarized in {history_summarized_time:.2f} seconds")

        return True

    except Exception as e:
        logging.error(f"An error occurred while summarizing messages: {e}", exc_info=True)
        return False    

def create_graph_chain(model, graph):
    try:
        logging.info(f"Graph QA Chain using LLM model: {model}")

        cypher_llm,model_name = get_llm(model)
        qa_llm,model_name = get_llm(model)
        graph_chain = GraphCypherQAChain.from_llm(
            cypher_llm=cypher_llm,
            qa_llm=qa_llm,
            validate_cypher= True,
            graph=graph,
            # verbose=True, 
            allow_dangerous_requests=True,
            return_intermediate_steps = True,
            top_k=3
        )

        logging.info("GraphCypherQAChain instance created successfully.")
        return graph_chain, qa_llm, model_name

    except Exception as e:
        logging.error(f"An error occurred while creating the GraphCypherQAChain instance. : {e}") 
        raise

def get_graph_response(graph_chain, question):
    try:
        cypher_res = graph_chain.invoke({"query": question})
        
        response = cypher_res.get("result")
        cypher_query = ""
        context = []

        for step in cypher_res.get("intermediate_steps", []):
            if "query" in step:
                cypher_string = step["query"]
                cypher_query = cypher_string.replace("cypher\n", "").replace("\n", " ").strip() 
            elif "context" in step:
                context = step["context"]
        return {
            "response": response,
            "cypher_query": cypher_query,
            "context": context
        }
    
    except Exception as e:
        logging.error(f"An error occurred while getting the graph response : {e}")

def process_graph_response(model, graph, question, messages, history):
    try:
        graph_chain, qa_llm, model_version = create_graph_chain(model, graph)
        
        graph_response = get_graph_response(graph_chain, question)
        
        ai_response_content = graph_response.get("response", "Something went wrong")
        ai_response = AIMessage(content=ai_response_content)
        
        messages.append(ai_response)
        # summarize_and_log(history, messages, qa_llm)
        summarization_thread = threading.Thread(target=summarize_and_log, args=(history, messages, qa_llm))
        summarization_thread.start()
        logging.info("Summarization thread started.")
        metric_details = {"question":question,"contexts":graph_response.get("context", ""),"answer":ai_response_content}
        result = {
            "session_id": "", 
            "message": ai_response_content,
            "info": {
                "model": model_version,
                "cypher_query": graph_response.get("cypher_query", ""),
                "context": graph_response.get("context", ""),
                "mode": "graph",
                "response_time": 0,
                "metric_details": metric_details,
            },
            "user": "chatbot"
        }
        
        return result
    
    except Exception as e:
        logging.exception(f"Error processing graph response at {datetime.now()}: {str(e)}")
        return {
            "session_id": "",  
            "message": "Something went wrong",
            "info": {
                "model": model_version,
                "cypher_query": "",
                "context": "",
                "mode": "graph",
                "response_time": 0,
                "error": f"{type(e).__name__}: {str(e)}"
            },
            "user": "chatbot"
        }

def create_neo4j_chat_message_history(graph, session_id, write_access=True):
    """
    [ENG]: Creates and returns a Neo4jChatMessageHistory instance.
    [IDN]: Membuat dan return instance berupa Neo4jChatMessageHistory.
    """
    try:
        if write_access: 
            history = Neo4jChatMessageHistory(
                graph=graph,
                session_id=session_id
            )
            return history
        
        history = get_history_by_session_id(session_id)
        return history

    except Exception as e:
        logging.error(f"Error creating Neo4jChatMessageHistory: {e}")
        raise 

def get_chat_mode_settings(mode, settings_map=CHAT_MODE_CONFIG_MAP):
    default_settings = settings_map[CHAT_DEFAULT_MODE]
    try:
        chat_mode_settings = settings_map.get(mode, default_settings)
        chat_mode_settings["mode"] = mode
        
        logging.info(f"Chat mode settings: {chat_mode_settings}")
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise

    return chat_mode_settings
    
def QA_RAG(graph, model, question, document_names, session_id, mode, write_access=True):
    logging.info(f"Chat Mode: {mode}")

    history = create_neo4j_chat_message_history(graph, session_id, write_access)
    messages = history.messages

    user_question = HumanMessage(content = question)
    messages.append(user_question)

    if mode == CHAT_GRAPH_MODE:
        result = process_graph_response(model, graph, question, messages, history)
    else:
        chat_mode_settings = get_chat_mode_settings(mode=mode)
        document_names= list(map(str.strip, json.loads(document_names)))
        if document_names and not chat_mode_settings["document_filter"]:
            result =  {
                "session_id": "",  
                "message": "Please deselect all documents in the table before using this chat mode",
                "info": {
                    "sources": [],
                    "model": "",
                    "nodedetails": [],
                    "total_tokens": 0,
                    "response_time": 0,
                    "mode": chat_mode_settings["mode"],
                    "entities": [],
                    "metric_details": [],
                },
                "user": "chatbot"
            }
        else:
            result = process_chat_response(messages, history, question, model, graph, document_names,chat_mode_settings)

    result["session_id"] = session_id
    
    return result

def chat_interaction(
    model: str,
    human_messages: str,
    session_id: str,
    context: Optional[Dict] = None,
    diagnosis: bool = False,
    disease_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    [ENG]: Handle medical chat interactions with diagnosis and informational modes.
    [IDN]: Mengelola interaksi obrolan medis dengan mode diagnosis dan informasi.

    Args:
        model: LLM model instance (Groq or Diffbot)
        human_messages: Current user message
        session_id: Unique session identifier for chat history
        context: User context including age, name, weight, height (JSON)
        diagnosis: Boolean flag for diagnosis mode
        disease_context: Specific disease context for informational mode
        is_initial: bool = False
    
    Returns:
        Dict containing response, symptoms (if applicable), and chat metadata
    """
    try:
        llm, model_name = get_llm(model)
        chat_history = get_history_by_session_id(session_id)
        messages = chat_history.messages

        context_str = ""
        if context:
            context_str = (
                f"Informasi Pasien:\n"
                f"Nama: {context.get('name', 'Tidak disebutkan')}\n"
                f"Usia: {context.get('age', 'Tidak disebutkan')}\n"
                f"Berat Badan: {context.get('weight', 'Tidak disebutkan')} kg\n"
                f"Tinggi Badan: {context.get('height', 'Tidak disebutkan')} cm\n"
                f"Deskripsi: {context.get('description', 'Tidak ada deskripsi tambahan')}\n\n"
            )

        if diagnosis:
            system_prompt = (
                f"Anda adalah seorang dokter yang memberikan informasi kepada pasien penderita penyakit. "
                f"{context_str}"
                "Berikan penjelasan berdasarkan konteks penyakit pasien yang ada."
                "Apabila tidak ada konteks yang relevan, berikan informasi umum dan jangan membuat informasi baru."
                "Jelaskan dengan bahasa yang mudah dipahami."
                "Jangan lakukan penanganan, hanya berikan informasi."
                "Tidak perlu memperkenalkan diri Anda."
                "Jawab dalam maksimum 2 kalimat."
                "Jawab dalam bahasa Indonesia."
                f"Penyakit pasien: {disease_context}"
            )
        else:
            system_prompt = (
                f"Ada seorang pasien dengan, {context_str}"
                "Anda adalah seorang dokter yang sedang memberikan konsultasi."
                "Selalu tanyakan gejala lain yang dirasakan pasien."
                "Fokus pada pengumpulan informasi yang relevan untuk diagnosis."
                "Jangan sebut nama penyakit apapun."
                "Jawab dalam kurang dari 14 kata."
            )

        if not messages:
            chat_history.add_message(SystemMessage(content=system_prompt))
        
        chat_history.add_message(HumanMessage(content=human_messages))

        if not diagnosis:
            # Get previous AI message if it exists
            previous_ai_message = None
            if len(messages) >= 2 and isinstance(messages[-1], HumanMessage) and isinstance(messages[-2], AIMessage):
                previous_ai_message = messages[-2].content
            
            extraction_prompt = (
                "Analisis pesan pasien berikut dan ekstrak apabila terdapat keluhan medis atau gejala dalam format JSON.\n"
                
                # Add context from previous AI message if available
                f"{'Pertanyaan sebelumnya dari dokter: ' + previous_ai_message if previous_ai_message else ''}\n\n"
                
                "Berikan hasil dalam format JSON:\n"
                "{\n"
                '  "gejala": ["pusing", "batuk", "lemas"]\n'
                "}\n\n"

                "Contoh:\n"
                "Pertanyaan: 'Apakah Anda merasa pusing atau mual?'\n"
                "Pesan Pasien: 'Engga, tetapi saya kesulitan ereksi'\n"
                "Jawaban: { \"gejala\": [\"kesulitan ereksi\"]}\n\n"

                "Contoh:\n"
                "Pertanyaan: 'Apakah Anda merasa pusing atau mual?'\n"
                "Pesan Pasien: 'Iya'\n"
                "Jawaban: { \"gejala\": [\"pusing\", \"mual\"]}\n\n"
                
                "Catatan penting:\n"
                "- Jika pasien menjawab 'ya', 'iya', 'ada', 'betul', dll. terhadap pertanyaan tentang gejala tertentu, ekstrak gejala tersebut\n"
                "- Gunakan konteks dari pertanyaan sebelumnya untuk memahami jawaban pasien yang singkat\n"
                "- Apabila tidak ada gejala, isi dengan { \"gejala\": []}\n\n"
                
                "Hanya jawab dengan format JSON.\n"
                f"Pesan pasien: {human_messages}"
            )
            
            symptom_response = llm.invoke([HumanMessage(content=extraction_prompt)])
            print(symptom_response)
            
            if isinstance(symptom_response, AIMessage):
                extracted_symptoms = symptom_response.content.strip()
            else:
                extracted_symptoms = "{}"

        # Get llm response for the main conversation
        chat_response = llm.invoke(messages)
        total_tokens = get_total_tokens(chat_response, llm)
        
        # Process symptoms if in consultation mode
        symptoms_summary = None
        if not diagnosis and extracted_symptoms:
            try:
                symptoms_summary = json.loads(extracted_symptoms)
            except json.JSONDecodeError:
                logging.warning("Failed to parse symptom extraction JSON")
                symptoms_summary = {
                    "gejala": [],
                }

        # Save AI response to history
        chat_history.add_message(AIMessage(content=chat_response.content))

        return {
            "session_id": session_id,
            "message": chat_response.content,
            "symptoms_summary": symptoms_summary,
            "timestamp": datetime.now().isoformat(),
            "info": {
                "model": model_name,
                "total_tokens": total_tokens,
                "response_time": 0,
            },
            "user": "chatbot"
        }

    except Exception as e:
        logging.error(f"Error in chat_interaction: {str(e)}")
        raise Exception(f"Failed to process chat interaction: {str(e)}")
    
def initial_greeting(session_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    [ENG]: Generate the initial greeting message for medical consultation.
    [IDN]: Menghasilkan pesan sambutan awal untuk konsultasi medis.
    
    Args:
        session_id: Unique session identifier
        context: Optional dictionary containing patient information
        
    Returns:
        Dict containing the greeting response
    """
    patient_name = context.get('name', 'Bapak/Ibu') if context else 'Bapak/Ibu'
    greeting_prompt = (
        f"Selamat datang {patient_name}, saya Dr. DTETI. "
        "Apa yang bisa saya bantu? "
        "Mohon ceritakan keluhan yang Anda rasakan saat ini."
    )
    
    return {
        "session_id": session_id,
        "message": greeting_prompt,
        "info": {
                "model": "-",
                "total_tokens": 0,
                "response_time": 0,
            },
        "timestamp": datetime.now().isoformat(),
        "user": "chatbot"
    }

def check_if_chat_is_symptoms(human_messages: str, model: str, session_id: str) -> bool:
    """
    [ENG]: Check if the chat message contains symptoms extraction request.
    [IDN]: Periksa apakah pesan obrolan berisi permintaan ekstraksi gejala.
   
    Args:
        human_messages: User chat message
        model: Model name/identifier for the LLM
       
    Returns:
        Boolean flag indicating if the chat message is a symptom extraction request
    """
    try:
        llm, model_name = get_llm(model)
        
        # Get chat history using session_id
        chat_history = get_history_by_session_id(session_id)
        messages = chat_history.messages
        
        # Get previous AI message if it exists
        previous_ai_message = None
        if len(messages) >= 1 and isinstance(messages[-1], HumanMessage):
            # Find the most recent AI message before the current human message
            for i in range(len(messages)-2, -1, -1):
                if isinstance(messages[i], AIMessage):
                    previous_ai_message = messages[i].content
                    break
        
        system_prompt = (
            "Anda adalah seorang dokter yang ahli dalam mendeteksi apakah seseorang sedang "
            "membicarakan tentang gejala penyakit atau kondisi kesehatan mereka. "
            "Tugas Anda adalah menentukan apakah pesan yang diberikan berisi informasi "
            "tentang gejala, keluhan kesehatan, atau pertanyaan medis.\n\n"
            "PENTING: Jika dokter menanyakan tentang gejala dan pasien menjawab APAPUN yang "
            "mengonfirmasi gejala tersebut (seperti 'ya', 'iya', 'betul', 'ada', dll.) "
            "atau memberikan durasi/detail tentang gejala tersebut, maka ini "
            "harus dianggap sebagai pembicaraan tentang gejala.\n\n"
            "Berikan jawaban 'true' jika pesan berisi informasi tentang gejala atau "
            "keluhan kesehatan, atau jika pesan adalah respons terhadap pertanyaan dokter "
            "tentang gejala. Jawab 'false' jika tidak terkait dengan gejala atau keluhan kesehatan."
            "Contoh:\n"
            "Pertanyaan Sebelumnya: 'Apakah ada gejala lain yang anda rasakan?'\n"
            "Pesan Pasien: 'Engga, hanya itu saja'\n"
            "Jawaban: false"
        )
        
        context = ""
        if previous_ai_message:
            context = f"Pertanyaan dokter sebelumnya: {previous_ai_message}\n\n"
        
        user_prompt = (
            f"{context}"
            f"Pesan pasien: {human_messages}\n\n"
            f"Berdasarkan percakapan di atas, apakah pasien sedang membicarakan atau "
            f"merespons tentang gejala/keluhan kesehatan? Jawab hanya dengan 'true' atau 'false'."
        )
       
        if model.lower() == 'groq_llama3_70b' or 'openai' in model.lower():
            from langchain_core.messages import SystemMessage, HumanMessage
           
            prompt_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
           
            response = llm.invoke(prompt_messages)
            result = response.content.lower().strip()
        else:
            raise ValueError(f"Unsupported model: {model}")
       
        positive_indicators = ['true', 'ya', 'benar', 'iya', 'betul']
        
        logging.info(f"Symptom detection for '{human_messages}' with previous context '{previous_ai_message}': {result}")
        return any(indicator in result for indicator in positive_indicators)
       
    except Exception as e:
        logging.error(f"Error in check_if_chat_is_symptoms: {str(e)}")
        return True