import os
import logging
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def get_llm(model: str):
    """[ENG]: Retrieve the specified language model based on the model name.
    [IDN]: Mendapatkan model LLM yang ditentukan berdasarkan nama modelnya.
    Model Option: groq, diffbot"""
    model = model.lower().strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = os.environ.get(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    try:
        if "groq" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)
        
        elif "openai" in model:
            model_name, api_key = env_value.split(",")
            if "o3-mini" in model:
                llm= ChatOpenAI(
                api_key=api_key,
                model=model_name)
            else:
                llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0,
                )

    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name