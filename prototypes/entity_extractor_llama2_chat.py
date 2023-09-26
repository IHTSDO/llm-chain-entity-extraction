from llama_cpp import Llama
import json
import time
import re
import fhir_api
import multiprocessing
# import cProfile

# ANSI escape sequences for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

def colorize_text(text, replacements, color_code):
    for word in replacements:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"{color_code}{word}{COLOR_RESET}", text)
    return text

server_url = "http://localhost:8080"
valueset_url = "http://snomed.info/sct/900000000000207008/version/20230630?fhir_vs"
# valueset_url = "http://snomed.info/sct?fhir_vs=isa/138875005"

def match_snomed(term):
    # skip it term length is less than 3
    if len(term) < 3 or len(term) > 100:
        return None
    fhir_response = fhir_api.expand_valueset(server_url, valueset_url, term)
    best_match = None
    if (fhir_response and 'expansion' in fhir_response and 'contains' in fhir_response['expansion'] and len(fhir_response['expansion']['contains']) > 0):
        # Check if there is a case insensitive exact match in fhir_response['expansion']['contains']
        for item in fhir_response['expansion']['contains']:
            if item['display'].lower() == term.lower():
                best_match = item
                break
        # If there is no exact match, return the first match
        # print(fhir_response['expansion']['contains'])

        if not best_match:
            best_match = fhir_response['expansion']['contains'][0]
    return best_match


# MODEL_PATH = "/Users/alo/llm/llama2/llama-2-13b-chat/ggml-model-q4_0.bin"
MODEL_PATH = "/Users/yoga/llama-cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin"
# Avoid threading onto efficiency cores. Set to None for automatic.
N_THREADS = max(1, multiprocessing.cpu_count() - 4)
USE_GPU = True

if USE_GPU:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=True, n_gpu_layers=128,
                n_threads=N_THREADS, use_mlock=True)
else:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False,
                n_threads=N_THREADS, use_mlock=True)

def simplify(term):
    prompts = [ { "role": "system", "content": """You are a clinical entity simplifier. Respond with simpler forms of the terms provided by the user.
The goal is to make the terms easier to match with SNOMED.
SNOMED is a clinical terminology that does not use plurals or other non-essential words. Remove plurals, and other non-essential words.
Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use."""},
                {"role":"user", "content":"pain in hands"},
                {"role":"assistant", "content":"pain in hand"},
                {"role":"user", "content":"multiple vesicular lesions"},
                {"role":"assistant", "content":"vesicular lesion"},
                {"role":"user", "content":term} ]
    response = llm.create_chat_completion(prompts, max_tokens=2048, temperature=0)
    return response["choices"][0]["message"]["content"]

def generalise(term):
    prompts = [ { "role": "system", "content": """Your task is to answer in a consistent style. Provide a new short an concise medical term without the specificity of the original term. Do not include any commentary or explanation in your response.
The goal is to make the terms easier to match with SNOMED.
SNOMED is a clinical terminology that does not use plurals or other non-essential words. Remove plurals, and other non-essential words.
Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use."""},
                {"role":"user", "content":"large pleural effusion"},
                {"role":"assistant", "content":"pleural effusion"},
                {"role":"user", "content":"intermittent asthma"},
                {"role":"assistant", "content":"asthma"},
                {"role":"user", "content":term} ]
    response = llm.create_chat_completion(prompts, max_tokens=2048, temperature=0)
    return response["choices"][0]["message"]["content"]

def rate_accuracy(term, snomed_code_name, context):
    prompts = [ {"role":"system", "content": """You are a clinical code verifier. You read clinical phrases and assess whether \
they match with the selected clinical concepts from SNOMED. You will be given a clinical term and a SNOMED code that was selected as a representation of its meaning by a student.
SNOMED is a clinical terminology that does not use plurals or other non-essential words. 
You will also be provided with the full clinical phrase that contains the clinical term.
Provide an assessment of how accurate the representation is on a scale from 1 to 5, where 1 means no meaningful relationship and 5 means identical meaning.                 
Do not include any commentary or explanation in your response."""},
                {"role":"user", "content":"Term: asthma\nSNOMED code: 195967001 |Asthma|\nContext: The patient has asthma and diabetes."},
                {"role":"assistant", "content":"5"},
                {"role":"user", "content":"Term: secondary renal hypertension\nSNOMED code: 38341003 |Hypertensive disorder, systemic arterial|\nContext: An 85-years-old woman has hip fracture and secondary renal hypertension."},
                {"role":"assistant", "content":"3"},
                {"role":"user", "content":"Term: vesicular skin rashes\nSNOMED code: 2904007 |Male infertility (disorder)|\nContext: A 15-year-old man with fever, cough and vesicular skin rashes that started a week ago was admitted."},
                {"role":"assistant", "content":"1"},
                {"role":"user", "content":f"Term: {term}\nSNOMED code: {snomed_code_name}\nContext: {context}"} ]
    response = llm.create_chat_completion(prompts, max_tokens=2048, temperature=0)
    return response["choices"][0]["message"]["content"]

extract_prompts = [
    { "role": "system", "content": """You are a clinical entity extractor. Report results as a JSON array of objects. \
Review the clinical notes provided by the user and extract all mentions of symptoms, diagnoses, procedures, and medications. \
Each clinical note is independent. \
Don't include demographics, commentary or clarification, only entities and the requested properties. \
Provide the following information for each entity:
 - text: text of the entity
 - type: type of clinical entity (symptom, diagnosis, procedure, medication, etc.)
 - context: present or absent"""},
    {"role":"user", "content":"A 76-year-old man with a history of chronic back pain presented with dizziness and altered mental status. Laboratory evaluation identified anion-gap metabolic acidosis."},
    {"role":"assistant", "content":'[{"text":"chronic back pain"}, {"text":"dizziness"}, {"text":"altered mental status"}, {"text":"anion-gap metabolic acidosis"}]'},
    {"role":"user", "content":""}
    ]

# Run the LLM on a test prompt to make sure it's awake
start_time = time.time()
print('Running initialisation test')
response = llm.create_chat_completion([
    { "role": "system", "content": "You talk."},
    {"role":"user", "content":"Say hi."}
    ], max_tokens=4, temperature=0)
print(repr(response))
print(COLOR_BLUE, "Elapsed time: ", time.time() - start_time, "Finish reason:", response["choices"][0]["finish_reason"], "Total tokens:", response["usage"]["total_tokens"], COLOR_RESET)


# Open the file in read mode
with open("clinical_text.txt", "r") as file:
    # Iterate over each line in the file
    for line in file:
        print("----------------------------------------------")
        print(line.strip())
        extract_prompts[-1]["content"] = line
        start_time = time.time()
        response = llm.create_chat_completion(extract_prompts, max_tokens=2048, temperature=0)
        print(COLOR_BLUE, "Elapsed time: ", time.time() - start_time, "Finish reason:", response["choices"][0]["finish_reason"], "Total tokens:", response["usage"]["total_tokens"], COLOR_RESET)
        json_text = response["choices"][0]["message"]["content"]
        pattern = r'\[.*\]'
        match = re.search(pattern, json_text, re.DOTALL)
        if match:
            json_array = re.search(pattern, json_text, re.DOTALL).group()
            try:
                results = json.loads(json_array)
            except json.decoder.JSONDecodeError:
                print(COLOR_RED, "Invalid or malformed JSON:", json_array, COLOR_RESET)
                continue
            if type(results[0]) == str:
                results = [{"text": text} for text in results]
            # Generate colorized text for output
            detectedTerms = []
            for entity in results:
                detectedTerms.append(entity["text"])
            # print(detectedTerms)
            resultString = colorize_text(line, detectedTerms, COLOR_GREEN)
            print(resultString.strip())
            # Match with SNOMED
            for entity in results:
                best_match = match_snomed(entity["text"])
                if best_match:
                    print(entity["text"], ":", COLOR_GREEN, best_match["code"],  best_match["display"], COLOR_RESET, end='')
                else:
                    simple =  simplify(entity["text"])
                    best_match = match_snomed(simple)
                    if best_match:
                        print(entity["text"],COLOR_YELLOW,"(", simple, ")", COLOR_RESET, ":", COLOR_GREEN, best_match["code"],  best_match["display"], COLOR_RESET, end='')
                    else:
                        general = generalise(entity["text"])
                        best_match = match_snomed(general)
                        if best_match:
                            print(entity["text"], ":", COLOR_YELLOW, f"( {simple}: {general} )", COLOR_GREEN, best_match["code"],  best_match["display"], COLOR_RESET, end='')
                        else:
                            print(entity["text"],COLOR_YELLOW, f"( {simple}: {general} )", COLOR_RESET, ":", COLOR_RED, "No match", COLOR_RESET, end='')
            
                if best_match:
                    accuracy = rate_accuracy(entity["text"], f'{best_match["display"]} |{best_match["code"]}|', line)
                    print(COLOR_BLUE, "Accuracy rating: ", accuracy, COLOR_RESET)
        else:
            print(COLOR_RED, "No entities detected", COLOR_RESET)
