import json 
import LLM 

with open("LLM_CONFIG/config.json", "r") as file:
    data = json.load(file)

def get_llm_obj(model_name):
    if data["LLM_MODEL"]["MAIN_MODEL"] == "llm_OpenAI":
        return LLM.OpenAILLM(model_name)
    if data["LLM_MODEL"]["MAIN_MODEL"] == "llm_Anthropic":
        return LLM.AnthropicLLM(model_name)
    
def get_response(question) -> str:
    obj = get_llm_obj(data["LLM_MODEL"]["INNER_MODEL"])
    return obj.get_response(question)