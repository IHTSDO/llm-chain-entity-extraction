"""Extracts clinical entities from free text using LLMs."""
import fhir_api
from prompts import *
import json
import time
import re
import argparse

DEFAULT_MODEL = 'openai'

# Get system arguments (api, model)
arg_parser = argparse.ArgumentParser(prog='LLM CT Entity Extractor', description='Extracts entities from clinical text using LLMs.')
arg_parser.add_argument('-a', '--api', default=DEFAULT_MODEL, help='the API to use (options: llama, openai, bard)')
arg_parser.add_argument('--model', help='the model to run, dependent upon API choice')
_args = arg_parser.parse_args()
llm_api = _args.api.lower()

# Conditionally import the chat completion function which uses the given API
if llm_api == 'llama':
    from completion_llama import create_chat_completion
    accuracy_prompts = accuracy_prompts_llama
elif llm_api == 'openai':
    from completion_openai import create_chat_completion
elif llm_api == 'bard':
    from completion_bard import create_chat_completion
else:
    raise NotImplemented(f'Please set the api argument to one of (llama, openai, bard). Got {llm_api}.')

"""
def create_chat_completion_wrapped(prompt, **kwargs):
    # Override any model parameters here, for example, setting temperature to 0:
    kwargs['temperature'] = kwargs.get('temperature', 0)
    return create_chat_completion(prompt, **kwargs)
create_chat_completion = create_chat_completion_wrapped
"""

# Constants for the kind of match
NO_MATCH, DIRECT_MATCH, SIMPLIFIED_MATCH, GENERALISED_MATCH, RESPELLED_MATCH = 'NO MATCH', 'DIRECT', 'SIMPLIFIED', 'GENERALISED', 'RESPELLED'

# ANSI escape sequences for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

def colorize_text(text, replacements, color_code):
    # space_sequence = r"(\.\,)?(\033\[0m)?\ "  # overwrite COLOR_RESET tags
    # (?=(
    for word in replacements:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"{color_code}{word}{COLOR_RESET}", text)
    return text

def display_color(line, entities, time_taken):
    print(colorize_text(line, entities.keys(), COLOR_GREEN))
    print(f'(overlapping matches not highlighted)\n{len(entities.keys())} entities found in {round(time_taken, 1)}s:')
    for key, value in entities.items():
        if value is None or value[2] == NO_MATCH:
            match, rating, *method_info = value
            print(f'{key} {COLOR_RED}No match{COLOR_RESET} {method_info}')
            continue
        
        match, rating, *method_info = value
        color = {5: COLOR_GREEN, 4: COLOR_YELLOW}.get(rating, COLOR_RED)

        print(f'{color}{key}{COLOR_RESET}: {COLOR_BLUE}{match["display"]} |{match["code"]}| {color}(confidence {rating}){COLOR_RESET} {method_info}')
    # Fail case where entities = {}
    if not entities:
        print(f'{COLOR_RED}Could not identify.{COLOR_RESET}')

    print(COLOR_BLUE, '---', COLOR_RESET)

server_url = "https://snowstorm-micro.nw.r.appspot.com"
# server_url = "https://snowstorm.ihtsdotools.org/fhir"
# server_url = "http://localhost:8080"
valueset_url = "http://snomed.info/sct?fhir_vs=isa/138875005"
# valueset_url = "http://snomed.info/sct/900000000000207008/version/20230630?fhir_vs"
# valueset_url = "http://snomed.info/sct?fhir_vs=isa/138875005"

def match_snomed(term, context=None):
    # skip it term length is less than 3
    if len(term) < 3 or len(term) > 100:
        return None
    fhir_response = fhir_api.expand_valueset(server_url, valueset_url, term)
    best_match = None
    # Check for valid response (whether terms were found or not)
    if (fhir_response and 'expansion' in fhir_response):
        # Non-empty responses only
        if 'contains' in fhir_response['expansion'] and len(fhir_response['expansion']['contains']) > 0:
            # Check if there is a case insensitive exact match in fhir_response['expansion']['contains']
            list_of_matches = fhir_response['expansion']['contains']
            # print([item['display'] for item in list_of_matches])
            for item in list_of_matches:
                if item['display'].lower() == term.lower():
                    best_match = item
                    break
            # If there is no exact match, return the best match from the top 5
            if not best_match:
                if len(list_of_matches) == 1:
                    best_match = list_of_matches[0]
                elif context is not None and context != term:
                    best_match = select_most_similar(f'{term}\nContext: {context}', list_of_matches)
                else:
                    best_match = select_most_similar(term, list_of_matches)
    else:
        print(fhir_response, '\n', COLOR_RED, 'Unable to get fhir response. Using blank codes.', COLOR_RESET, sep='')
        best_match = {'display': str.capitalize(term) + '*', 'code': 'Unavailable'}
    return best_match

def select_most_similar(term, list_of_matches):
    list_of_names = [match['display'] for match in list_of_matches]
    comma_separated_list_of_names = ', '.join(list_of_names)
    select_best_prompts[-1]['content'] = "Clinician's term: {}\nPossible matches from SNOMED: {}".format(term, comma_separated_list_of_names)
    response = create_chat_completion(select_best_prompts, max_tokens=16)

    if response in list_of_names:
        for match in list_of_matches:
            if match['display'].lower() == response.lower():
                return match
        else:
            return list_of_matches[0]
    else:
        return list_of_matches[0]

def rate(term, match, context):
    """Rate the accuracy of the assigned match to the term on a rating of 1 to 5, or 0 if the response is invalid."""
    # gpt-4 can follow the no explanations rule but the other models sometimes do not,
    # so we allow a longer response and extract the rating from it.
    max_tokens_for_rating = 1 if llm_api=='openai' else 256

    if term.lower() == match.lower():
        return 5
    # accuracy_prompts[-1]['content'] = "Clinician's term: {}\nSNOMED term: {}\nContext: {}".format(term, match, context)
    accuracy_prompts[-1]['content'] = "Clinician's term: \"{}\"\nSNOMED term: \"{}\"\nContext: \"{}\"".format(term, match, context)
    response = create_chat_completion(accuracy_prompts, max_tokens=max_tokens_for_rating).strip()
    if response in ('1', '2', '3', '4', '5'):
        return int(response)
    elif string := extract_first_digit_in_range_final(response): #match := re.match('[1-5](\.\d)?', response):
        print(COLOR_BLUE, f'Match response: {response[:40]} ...', COLOR_RESET, '->', string)
        return int(float(string) // 1)  # or math.floor
    else:
        print(COLOR_RED, f'Invalid rating response: {response}', COLOR_RESET)
        return 0

def extract_first_digit_in_range_final(s):
    match = re.search(r'(?<![.\d])([1-5](\.\d+)?)', s)
    return match.group(1) if match else None

def identify_language(lines):
    # Join lines together (up to first 12) in case certain lines are headings or comments
    # that are too short to determine the language from.
    text = ' // '.join(lines[:12])
    # Get a short piece of the text (up to 48 characters). Cut it off at a word or sentence boundary.
    text_extract = ' '.join(text[:48].split(' ')[:-1])
    # Use the extract from the text to detect the language.
    return create_chat_completion(from_prompt(language_prompts, text_extract), max_tokens=16).strip().lower()

def as_english(text, language_description):
    """Return the text translated to English (if required)."""
    # If the language isn't English, translate the text depending on the language detected.
    response = language_description
    if 'en' in response:
        pass
    elif 'es' in response or 'spanish' in response:
        text = create_chat_completion(from_prompt(translate_es_en_prompts, text))
        print(f'Language: {response} ({COLOR_BLUE}translating es -> en{COLOR_RESET})')
    else:
        text = create_chat_completion(from_prompt(translate_en_prompts, text))
        print(f'Language: {response} (unexpected response; {COLOR_BLUE}translating any language -> en{COLOR_RESET})')
    return text

def from_prompt(prompts, term):
    prompts[-1]['content'] = term
    return prompts

def identify(text):
    """Return the clinical entities in a clinical note or sample of free text."""
    # print(text)

    # Query the model for a chat completion that extracts entities from the text.
    json_text = create_chat_completion(from_prompt(extract_prompts, text))
    pattern = r'\[.*\]'
    # Search for a json array.
    match = re.search(pattern, json_text, re.DOTALL)
    if match:
        json_array = match.group()
        try:
            results = json.loads(json_array)
            response_terms = [result['text'] for result in results]
        except (json.decoder.JSONDecodeError, TypeError):
            print(COLOR_RED, "Invalid or malformed JSON:", json_array, COLOR_RESET)
            return {}
    else:
        return {}

    # Dictionary to assign each term a list of information about the match found, confidence and strategy used.
    term_results = {}

    for term in response_terms:
        term_results[term] = ['', 0, NO_MATCH]
        print(f'Matching: {term}'.ljust(80), end='\r')

        # Look for direct matches with SNOMED CT
        potential_match = match_snomed(term)
        if potential_match:
            rating = rate(term, potential_match['display'], text)
            term_results[term] = [potential_match, rating, DIRECT_MATCH]
            if rating > 3:
                continue
        else:
            # Always replace
            rating = 0

        # Attempt to simplify the term in order to improve on the initial match or find a match
        simple_term = create_chat_completion(from_prompt(simplify_prompts, term), max_tokens=16)
        term_results[term].append(simple_term)

        new_potential_match = match_snomed(simple_term, context=term)
        # Don't repeat queries to the LLM (implicitly: skip if both entries are None)
        if new_potential_match is not None and new_potential_match != potential_match:
            new_rating = rate(simple_term, new_potential_match['display'], term)
            if new_rating > rating:
                # Set the match or replace the previous match with the new match
                potential_match, rating = new_potential_match, new_rating
                term_results[term] = [potential_match, rating, SIMPLIFIED_MATCH, simple_term]
            if new_rating > 3:
                continue

        # Attempt to generalise the term
        general_term = create_chat_completion(from_prompt(generalise_prompts, term), max_tokens=16)
        term_results[term].append(general_term)

        # We search using the generalised term but still pick the best match based on our original term
        new_potential_match = match_snomed(general_term, context=term)
        if new_potential_match is not None and new_potential_match != potential_match:
            new_rating = rate(general_term, new_potential_match['display'], term)
            if new_rating > rating:
                # Set the match or replace the previous match with the new match
                potential_match, rating = new_potential_match, new_rating
                term_results[term] = [new_potential_match, new_rating, GENERALISED_MATCH, simple_term, general_term]
            # We use a lower threshold, since later steps should only be used if our current match is poor
            if new_rating >= 3:
                continue

    return term_results


def main():
    # Initialise the LLM we are using (if required)
    # Read the test cases (hide blank lines)
    with open("clinical_text_3.txt", "r") as file:
        stripped_lines = map(str.rstrip, file.readlines())
    # skip newlines and comments/titles    
    lines = [line for line in stripped_lines if line and not line.startswith('#')]
    
    start_time_all_cases = time.time()
    entities_per_line = []
    language = identify_language(lines)

    # Iterate over each line in the test cases
    for line in lines:
        print(COLOR_BLUE, line, COLOR_RESET, sep='')
        start_time = time.time()
        text = as_english(line, language)  # translate text in other languages to english
        entities = identify(text)  # identify entities
        entities_per_line.append(entities)
        display_color(text, entities, time.time() - start_time)

    print(COLOR_BLUE, time.time() - start_time_all_cases, 's', COLOR_RESET, sep='')

if __name__ == "__main__":
    main()
