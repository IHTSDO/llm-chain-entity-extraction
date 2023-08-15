import json
import time
import re
import fhir_api
import multiprocessing
import argparse

arg_parser = argparse.ArgumentParser(prog='CT Entity Extractor', description='Extracts entities from clinical text')
arg_parser.add_argument('-a', '--api', default='llama', help='the API to use (options: llama, openai, bard)')
arg_parser.add_argument('--model', help='the model to run')
arg_parser.add_argument('--gpu', help='the')

llm_api = arg_parser.parse_args().api.lower()

# Conditionally import the chat completion function which uses the given API
if llm_api == 'llama':
    from completion_llama import create_chat_completion
elif llm_api == 'openai':
    from completion_openai import create_chat_completion
elif llm_api == 'bard':
    from completion_bard import create_chat_completion
else:
    raise NotImplemented(f'Please set the api argument to one of (llama, openai, bard). Got {llm_api}.')


    

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
        # print(fhir_response['expansion']['contains'])

        if not best_match:
            best_match = fhir_response['expansion']['contains'][0]
    return best_match

def main():
    # Initialise the LLM we are using
    pass

if __name__ == "__main__":
    main()
