import openai
from nltk.stem.lancaster import LancasterStemmer
import json
import time
import re
import fhir_api

openai.api_key = "sk-________________________________________________"

# ANSI escape sequences for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

def colorize_text(text, replacements, color_code):
    for word in replacements:
        text = text.replace(word, f"{color_code}{word}{COLOR_RESET}")
    return text

# list models
models = openai.Model.list()

# print the accesssible text completion models
print(', '.join(str(model_data.id) for model_data in models.data))

MODEL = "gpt-4"

# Recommended for deterministic and foocused output that is 'more likely to be correct and efficient'.
TEMPERATURE = 0.2

server_url = "https://snowstorm.ihtsdotools.org/fhir"
# server_url = "http://localhost:8080"
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
        if not best_match:
            best_match = fhir_response['expansion']['contains'][0]
    return best_match

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
    response = openai.ChatCompletion.create(model=MODEL, messages=prompts, temperature=TEMPERATURE)
    return response["choices"][0]["message"]["content"]

def generalise(term):

    prompts = [ { "role": "system", "content": """You are a clinical entity simplifier.
                 The goal is to make the terms easier to match with SNOMED.
                 SNOMED is a clinical terminology that does not use plurals or other non-essential words. Remove plurals, and other non-essential words.
                 Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use."""},
                {"role":"user", "content":"large pleural effusion"},
                {"role":"assistant", "content":"pleural effusion"},
                {"role":"user", "content":"intermittent asthma"},
                {"role":"assistant", "content":"asthma"},
                {"role":"user", "content":term} ]
    response = openai.ChatCompletion.create(model=MODEL, messages=prompts, temperature=TEMPERATURE)
    return response["choices"][0]["message"]["content"]

""" You will also be provided with the full clinical phrase that contains the clinical term.
SNOMED is a clinical terminology that does not use plurals or other non-essential words.
Provide an assessment of how accurate the representation is on a scale from 1 to 5, where 1 means no meaningful relationship and 5 means identical meaning.
"""
"""You are a clinical language evaluator.
You will be given two clinical terms and you need to assess how closely they are related on a scale from 1 to 5, where 1 means no meaningful relationship and 5 means identical meaning.
Do not include any commentary or explanation in your response."""
"""

                {"role":"user", "content":"Term 1: asthma\nTerm 2: Asthma"},
                {"role":"assistant", "content":"5"},
                {"role":"user", "content":"Term 1: mild fever\nTerm 2: Fever"},
                {"role":"assistant", "content":"4"},
                {"role":"user", "content":"Term 1: secondary renal hypertension\nTerm 2: Hypertensive disorder, systemic arterial"},
                {"role":"assistant", "content":"3"},
                {"role":"user", "content":"Term 1: leg cramp\nTerm 2: Pain"},
                {"role":"assistant", "content":"2"},
                {"role":"user", "content":"Term 1: vesicular skin rashes\nTerm 2: Male infertility"},
                {"role":"assistant", "content":"1"},"""

def rate_accuracy(term, snomed_term):
    prompts = [ {"role":"system", "content": """You are clinical expert, that compares terms doctors write in clinical notes with SNOMED CT terms selected to represent the same meaning.
You will be given two clinical terms and you need to assess how closely they are related on a scale from 1 to 5, where 1 means no meaningful relationship and 5 means identical meaning.
Do not include any commentary or explanation in your response.
"""}, 
                {"role":"user", "content":f"Clinician's term: {term}\nSnomed term: {snomed_term}"}]
    response = openai.ChatCompletion.create(model=MODEL, messages=prompts, temperature=TEMPERATURE)
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

"""start_time = time.time()
print('Running initialisation test')
response = llm.create_chat_completion([
    { "role": "system", "content": "You talk."},
    {"role":"user", "content":"Say hi."}
    ], max_tokens=4, temperature=0)
print(repr(response))
print(COLOR_BLUE, "Elapsed time: ", time.time() - start_time, "Finish reason:", response["choices"][0]["finish_reason"], "Total tokens:", response["usage"]["total_tokens"], COLOR_RESET)
"""

# Open the file in read mode
with open("clinical_text.txt", "r") as file:
    # Iterate over each line in the file
    for line in file:
        print("----------------------------------------------")
        print(line.strip())
        extract_prompts[1]["content"] = "Extract clinical entities form this text:\n" + line
        start_time = time.time()
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=extract_prompts, temperature=TEMPERATURE)
        print(COLOR_BLUE, "Elapsed time: ", time.time() - start_time, "Finish reason:", response["choices"][0]["finish_reason"], "Total tokens:", response["usage"]["total_tokens"], COLOR_RESET)
        json_text = response["choices"][0]["message"]["content"]
        pattern = r'\[.*\]'
        match = re.search(pattern, json_text, re.DOTALL)
        if match:
            json_array = re.search(pattern, json_text, re.DOTALL).group()
            results = json.loads(json_array)
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
                            print(entity["text"],COLOR_YELLOW, f"( {simple}: {general} )", COLOR_RESET, ":", COLOR_RED, "No match", COLOR_RESET)
                            continue
            
                if best_match:
                    if entity["text"].lower() == best_match["display"].lower():
                        print(COLOR_BLUE, "(Identical)", COLOR_RESET)
                    else:
                        accuracy = rate_accuracy(entity["text"], best_match["display"])
                        print(COLOR_BLUE, f'(Accuracy rating: {accuracy})', COLOR_RESET)
                
        else:
            print(COLOR_RED, "No entities detected", COLOR_RESET)