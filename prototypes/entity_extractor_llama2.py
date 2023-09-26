from llama_cpp import Llama
from nltk.stem.lancaster import LancasterStemmer
import fhir_api
import multiprocessing

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
        if not best_match:
            best_match = fhir_response['expansion']['contains'][0]
    return best_match


# llm = Llama(model_path="/Users/alo/llm/llama.cpp/models/13B/ggml-model-q4_0.bin", verbose=False)
# llm = Llama(model_path="/Users/alo/llm/llama.cpp/models/vicuna-13b-v1.3/ggml-model-q4_0.bin", n_ctx=2048, verbose=False)
# llm = Llama(model_path="/Users/alo/llm/llama2/llama-2-13b-chat/ggml-model-q4_0.bin", n_ctx=2048, verbose=False)
MODEL_PATH = "/Users/yoga/llama-cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin"
EFFICIENCY_CORES = 4
USE_GPU = True

if USE_GPU:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=True, n_gpu_layers=128,
                n_threads=max(1, multiprocessing.cpu_count() - EFFICIENCY_CORES), use_mlock=True)
else:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False,
                n_threads=max(1, multiprocessing.cpu_count() - EFFICIENCY_CORES), use_mlock=True)

#llm = Llama(model_path="/Users/alo/llm/llama.cpp/models/vicuna-13b-v1.3/ggml-model-q4_0.bin", n_ctx=2048, verbose=False)

def simplify(term):
    q = """
    I need to search these clinical phrases in SNOMED, and I know they don't match exactly as they are now. Please generate an equivalent the clinical phrase to facilitate matching. Simplify by removing plurals, and other non-essential words. Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use.
    Q: right hip fractures
    A: hip fracture
    Q: has diabetes mellitus
    A: diabetes mellitus
    Q: several scars
    A: scar
    Q: catastroophic arterial hypertension
    A: arterial hypertension
    """
    prompt = q + "\nQ: " + term + "\n"
    output = llm(prompt, max_tokens=512, stop=["Q:", "\n"], echo=True, temperature=1.0)
    a = output['choices'][0]['text']
    # remove the question
    a = a[len(prompt):]
    return a[3:]

def generalize(term):
    q = """
    Please analyze the following medical text and generalize specific medical terms to more broad categories. Your task is to provide a new short an concise medical term without the specificity of the original term. Do not include any commentary or explanation in your response.
    Q: back pain
    A: pain
    Q: intermitent asthma
    A: asthma
    Q: massive hemoptysis
    A: hemoptysis
    Q: large pleural effusion
    A: pleural effusion
    """
    prompt = q + "\nQ: " + term + "\n"
    output = llm(prompt, max_tokens=512, stop=["Q:", "\n"], echo=True, temperature=1.0)
    a = output['choices'][0]['text']
    # remove the question
    a = a[len(prompt):]
    return a[3:]

st = LancasterStemmer()

q = """
Review the following clinical notes and extract all mentions of symptoms, diagnoses, procedures, and medications. Please make sure to exclude any severity or laterality modifiers from your extractions. Include also any medical acronyms detected. The goal is to establish a clear list of the patient's signs and symptoms, the diagnoses made, any procedures performed or planned, and medications prescribed or taken, while omitting details related to the severity of symptoms or diagnoses and the side of the body affected.
Q: A 44-year-old woman was evaluated in the rheumatology clinic of this hospital because of proximal muscle weakness and myalgia. He has a history of liver nodules.
A: [ 'muscle weakness', 'myalgia', 'liver nodules' ]
Q: A 76-year-old man with a history of chronic back pain presented with dizziness and altered mental status. Laboratory evaluation identified anion-gap metabolic acidosis.
A: [ 'chronic back pain', 'dizziness', 'altered mental status', 'anion-gap metabolic acidosis' ]
Q: A 47-year-old woman with malignant melanoma was found to have bilateral pulmonary nodules, hilar and mediastinal lymphadenopathy, and left upper lobe infiltrate. An MRI was performed.
A: [ 'malignant melanoma', 'pulmonary nodules', 'hilar and mediastinal lymphadenopathy', 'upper lobe infiltrate', 'Magnetic Resonance Imaging' ]
"""
# Open the file in read mode
with open("clinical_text.txt", "r") as file:
    # Iterate over each line in the file
    for line in file:
        line = "\nQ: " + line + "\n"
        # Process each line
        print(COLOR_BLUE + line + COLOR_RESET)
        prompt = q + line
        output = llm(prompt, max_tokens=2048, stop=["Q:", "\n"], echo=False, temperature=1.0)
        a = output['choices'][0]['text']
        # remove the question
        # a = a[len(prompt):]
        results = eval(a[3:])
        resultString = line[3:]
        resultString = colorize_text(resultString, results, COLOR_GREEN)
        print(resultString)
        for result in results:
            snomed_match = match_snomed(result)
            if snomed_match:
                print(COLOR_GREEN, result, COLOR_BLUE, snomed_match['code'], snomed_match['display'], COLOR_RESET)
            else:
                simplified = simplify(result)
                if simplified != result:
                    snomed_match = match_snomed(simplified)
                if snomed_match:
                    print(COLOR_GREEN, result, COLOR_YELLOW, simplified, COLOR_BLUE, snomed_match['code'], snomed_match['display'], COLOR_RESET)
                else:
                    single_word = generalize(result)
                    if single_word != simplified:
                        snomed_match = match_snomed(single_word)
                    if snomed_match:
                        print(COLOR_GREEN, result, COLOR_YELLOW, simplified, single_word, COLOR_BLUE, snomed_match['code'], snomed_match['display'], COLOR_RESET)
                    else:
                        print(COLOR_RED, result, COLOR_YELLOW, simplified, single_word, COLOR_RESET)
