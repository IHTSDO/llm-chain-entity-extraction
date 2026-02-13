"""Extracts clinical entities from free text using LLMs."""
import fhir_api
from prompts import *
import json
import time
import re
import argparse
import os

DEFAULT_MODEL = 'openai'

# Get system arguments (api, model)
arg_parser = argparse.ArgumentParser(prog='LLM CT Entity Extractor', description='Extracts entities from clinical text using LLMs.')
arg_parser.add_argument('-a', '--api', default=DEFAULT_MODEL, help='the API to use (options: llama, openai, bard)')
arg_parser.add_argument('--model', help='the model to run, dependent upon API choice. Path to model in Llama2.0, model name in OpenAI, or model name in BARD.')
arg_parser.add_argument('--sentences', help='path to a set of free text clinical sentences to encode with SNOMED CT.')
arg_parser.add_argument('--n', type=int, help='optional limit for number of non-empty, non-comment lines to process')
arg_parser.add_argument('--mode', choices=['verbose', 'summary'], default='verbose', help='output mode: verbose (default) or summary')
_args = arg_parser.parse_args()
llm_api = _args.api.lower()
output_mode = _args.mode.lower()

if _args.model:
    os.environ['OPENAI_MODEL'] = _args.model


def ensure_openai_api_key():
    try:
        with open('openai.key', 'r') as file:
            api_key = file.read().strip()
            if not api_key:
                raise ValueError(
                    'openai.key is empty. Paste your OpenAI API key in that file.'
                )
            if not api_key.startswith('sk-'):
                raise ValueError(
                    "Invalid key format in openai.key. It should start with 'sk-'."
                )
            os.environ['OPENAI_API_KEY'] = api_key
    except FileNotFoundError as exc:
        raise ValueError(
            'Missing openai.key. Create it in the project root and paste your OpenAI API key there.'
        ) from exc


# Conditionally import the chat completion function which uses the given API
if llm_api == 'llama':
    from completion_llama import initialize_model
    from completion_llama import create_chat_completion
    create_chat_completion_json = None
    accuracy_prompts = accuracy_prompts_llama
    initialize_model(_args.model)
elif llm_api == 'openai':
    ensure_openai_api_key()
    from completion_openai import create_chat_completion
    from completion_openai import create_chat_completion_json
elif llm_api == 'bard':
    from completion_bard import create_chat_completion
    create_chat_completion_json = None
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
NO_MATCH, DIRECT_MATCH, PREFIX_MATCH, SIMPLIFIED_MATCH, GENERALISED_MATCH, RESPELLED_MATCH = 'NO MATCH', 'DIRECT', 'PREFIX', 'SIMPLIFIED', 'GENERALISED', 'RESPELLED'

# ANSI escape sequences for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
OK_MIN_RATING = 3
PROGRESS_INLINE = os.environ.get("OPENAI_DEBUG_CACHE", "").strip().lower() not in {"1", "true", "yes"}

ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["rating"],
    "additionalProperties": False,
}

def colorize_text(text, replacements, color_code):
    # space_sequence = r"(\.\,)?(\033\[0m)?\ "  # overwrite COLOR_RESET tags
    # (?=(
    for word in replacements:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"{color_code}{word}{COLOR_RESET}", text)
    return text

def progress_print(message):
    if output_mode == 'summary':
        return
    print(message.ljust(80), end='\r' if PROGRESS_INLINE else '\n')

def format_attempts_for_display(original_term, value):
    prefix = value[3] if len(value) > 3 else ''
    simplified = value[4] if len(value) > 4 else ''
    generalised = value[5] if len(value) > 5 else ''
    return {
        'direct': original_term,
        'prefix': prefix or '(empty)',
        'simplified': simplified or '(empty)',
        'generalised': generalised or '(empty)',
    }

def is_exportable_match(value, min_rating=OK_MIN_RATING):
    return (
        value is not None
        and len(value) >= 3
        and value[2] != NO_MATCH
        and isinstance(value[1], int)
        and value[1] >= min_rating
    )

def exportable_entities(entities, min_rating=OK_MIN_RATING):
    return {
        term: value
        for term, value in entities.items()
        if is_exportable_match(value, min_rating=min_rating)
    }

def sentence_score_counts(entities, min_rating=OK_MIN_RATING):
    ok_count = len(exportable_entities(entities, min_rating=min_rating))
    total = len(entities)
    return ok_count, total

def rating_to_color(value):
    if value is None or not isinstance(value, list) or len(value) < 2:
        return COLOR_RED
    rating = value[1]
    if isinstance(rating, int):
        if rating >= 4:
            return COLOR_GREEN
        if rating == 3:
            return COLOR_YELLOW
    return COLOR_RED

def colorize_text_by_rating(text, entities):
    # Longer terms first to reduce partial-overlap replacements.
    for term in sorted(entities.keys(), key=len, reverse=True):
        color_code = rating_to_color(entities.get(term))
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(lambda m: f"{color_code}{m.group(0)}{COLOR_RESET}", text)
    return text

def display_color(line, entities, time_taken):
    print(colorize_text_by_rating(line, entities))
    print(f'(overlapping matches not highlighted)\n{len(entities.keys())} entities found in {round(time_taken, 1)}s:')
    for key, value in entities.items():
        attempts = format_attempts_for_display(key, value)
        if value is None or len(value) < 3 or value[2] == NO_MATCH:
            print(f'{key} {COLOR_RED}No match{COLOR_RESET} attempts={attempts}')
            continue
        
        match, rating, *method_info = value
        color = COLOR_GREEN if rating >= 4 else (COLOR_YELLOW if rating == 3 else COLOR_RED)

        print(f'{color}{key}{COLOR_RESET}: {COLOR_BLUE}{match["display"]} |{match["code"]}| {color}(confidence {rating}){COLOR_RESET} {method_info} attempts={attempts}')
    # Fail case where entities = {}
    if not entities:
        print(f'{COLOR_RED}Could not identify.{COLOR_RESET}')
    else:
        ok_count, total = sentence_score_counts(entities, min_rating=OK_MIN_RATING)
        score_pct = round((ok_count / total) * 100, 1) if total else 0.0
        print(f'Sentence time: {round(time_taken, 1)}s | Sentence score (OK={OK_MIN_RATING}-5): {ok_count}/{total} ({score_pct}%)')

    print(COLOR_BLUE, '---', COLOR_RESET)

def display_summary(line, entities, time_taken):
    ok_count, total = sentence_score_counts(entities, min_rating=OK_MIN_RATING)
    score_pct = round((ok_count / total) * 100, 1) if total else 0.0
    print(colorize_text_by_rating(line, entities))
    print(f'Sentence score (green+yellow/total): {ok_count}/{total} ({score_pct}%) | time: {round(time_taken, 1)}s')
    print(COLOR_BLUE, '---', COLOR_RESET)

server_url = "https://implementation-demo.snomedtools.org/snowstorm-lite/fhir"
# server_url = "http://localhost:8080"
valueset_url = "http://snomed.info/sct?fhir_vs=isa/138875005"
# valueset_url = "http://snomed.info/sct/900000000000207008/version/20230630?fhir_vs"
# valueset_url = "http://snomed.info/sct?fhir_vs=isa/138875005"

def match_snomed(term, context=None):
    progress_print(f'Searching: {term}')
    # skip it term length is less than 3
    if len(term) < 3 or len(term) > 100:
        return None

    def normalize_for_exact(s):
        # Normalize minor punctuation/casing differences for exact display checks.
        return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

    # Prefer non-fuzzy retrieval first for direct matching.
    fhir_response = fhir_api.expand_valueset(server_url, valueset_url, term, fuzzy_match=False)
    if not (fhir_response and 'expansion' in fhir_response and fhir_response['expansion'].get('contains')):
        # Fallback to fuzzy if strict search returns no candidates.
        fhir_response = fhir_api.expand_valueset(server_url, valueset_url, term)

    best_match = None
    # Check for valid response (whether terms were found or not)
    if (fhir_response and 'expansion' in fhir_response):
        # Non-empty responses only
        if 'contains' in fhir_response['expansion'] and len(fhir_response['expansion']['contains']) > 0:
            # Check if there is a case insensitive exact match in fhir_response['expansion']['contains']
            list_of_matches = fhir_response['expansion']['contains']
            normalized_term = normalize_for_exact(term)
            for item in list_of_matches:
                if normalize_for_exact(item['display']) == normalized_term:
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

def tokenize_for_prefix(term):
    return [token for token in re.split(r'[^a-zA-Z0-9]+', term.lower()) if token]

def build_prefix_query(tokens, prefix_len=3):
    return [token[:prefix_len] if len(token) > prefix_len else token for token in tokens]

def prefix_query_variants(prefixes):
    # Full prefix first, then progressively drop the left-most prefix.
    return [' '.join(prefixes[i:]) for i in range(len(prefixes)) if prefixes[i:]]

def match_snomed_by_prefix(term, context=None, prefix_len=3):
    tokens = tokenize_for_prefix(term)
    if len(tokens) < 2:
        return None, None

    prefixes = build_prefix_query(tokens, prefix_len=prefix_len)
    variants = prefix_query_variants(prefixes)

    for variant in variants:
        fhir_response = fhir_api.expand_valueset(
            server_url,
            valueset_url,
            variant,
            fuzzy_match=False,
        )
        if not (fhir_response and 'expansion' in fhir_response):
            continue
        contains = fhir_response['expansion'].get('contains', [])
        if not contains:
            continue

        if len(contains) == 1:
            return contains[0], variant
        if context is not None and context != term:
            return select_most_similar(f'{term}\nContext: {context}', contains), variant
        return select_most_similar(term, contains), variant

    return None, None

def select_most_similar(term, list_of_matches):
    term = clean_string(term)
    progress_print('Selecting best match by similarity')
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

def _json_text_from_response(text):
    text = text.strip()
    if not text:
        return None
    # Fast path for valid JSON.
    if text[0] in ('{', '['):
        return text
    # Recover first JSON object/array if the model included extra text.
    obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    if obj_match:
        return obj_match.group(0)
    arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if arr_match:
        return arr_match.group(0)
    return None

def _load_json_response(text):
    candidate = _json_text_from_response(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.decoder.JSONDecodeError:
        return None

def _with_json_instruction(prompts, schema):
    prompts_copy = [dict(message) for message in prompts]
    schema_str = json.dumps(schema, separators=(',', ':'))
    prompts_copy.append(
        {
            "role": "user",
            "content": (
                "Return only valid JSON that matches this schema exactly. "
                f"Schema: {schema_str}"
            ),
        }
    )
    return prompts_copy

def create_structured_response(prompts, schema, schema_name, max_tokens=256, use_cache=True):
    if create_chat_completion_json is not None:
        try:
            response_json = create_chat_completion_json(
                prompts,
                schema=schema,
                schema_name=schema_name,
                max_tokens=max_tokens,
                use_cache=use_cache,
            )
            if isinstance(response_json, (dict, list)):
                return response_json
        except Exception:
            pass

    # Fallback path for providers/models without native schema support.
    text = create_chat_completion(
        _with_json_instruction(prompts, schema),
        max_tokens=max_tokens,
        use_cache=use_cache,
    ).strip()
    return _load_json_response(text)

def heuristic_simplify_term(term):
    """Fallback simplification when the model returns empty or unhelpful output."""
    t = clean_string(term).lower()
    if not t:
        return ''

    # Example: "eye redness" -> "red eye"
    m = re.match(r'^([a-z0-9-]+)\s+redness$', t)
    if m:
        return f'red {m.group(1)}'

    # Example: "redness of eye" -> "red eye"
    m = re.match(r'^redness of (?:the )?([a-z0-9-]+)$', t)
    if m:
        return f'red {m.group(1)}'

    return t

def rate(term, match, context):
    progress_print(f'Rating: {term} vs. {match}')
    """Rate the accuracy of the assigned match to the term on a rating of 1 to 5, or 0 if the response is invalid."""
    # gpt-4 can follow the no explanations rule but the other models sometimes do not,
    # so we allow a longer response and extract the rating from it.
    max_tokens_for_rating = 16 if llm_api=='openai' else 256

    if term.lower() == match.lower():
        return 5
    accuracy_prompts[-1]['content'] = "Clinician's term: \"{}\"\nSNOMED term: \"{}\"\nContext: \"{}\"".format(term, match, context)

    response_json = create_structured_response(
        accuracy_prompts,
        schema=RATING_SCHEMA,
        schema_name="entity_match_rating",
        max_tokens=max_tokens_for_rating,
    )
    if isinstance(response_json, dict):
        rating = response_json.get("rating")
        if isinstance(rating, int) and 1 <= rating <= 5:
            return calibrate_rating(term, match, rating)

    # Retry once with stricter wording in case the model ignored schema/instructions.
    strict_rating_prompts = [
        {
            "role": "system",
            "content": (
                "You are a strict classifier. Return JSON only using this shape: "
                "{\"rating\": <integer 1-5>}."
            ),
        },
        {
            "role": "user",
            "content": "Clinician's term: \"{}\"\nSNOMED term: \"{}\"\nContext: \"{}\"".format(term, match, context),
        },
    ]
    retry_json = create_structured_response(
        strict_rating_prompts,
        schema=RATING_SCHEMA,
        schema_name="entity_match_rating_retry",
        max_tokens=16,
    )
    if isinstance(retry_json, dict):
        retry_rating = retry_json.get("rating")
        if isinstance(retry_rating, int) and 1 <= retry_rating <= 5:
            return calibrate_rating(term, match, retry_rating)

    fallback = heuristic_rating(term, match)
    print(COLOR_RED, f'Invalid rating response for: {term} vs. {match} -> fallback {fallback}', COLOR_RESET)
    return fallback

def heuristic_rating(term, match):
    """Deterministic fallback when the model doesn't return a valid numeric score."""
    lhs = set(re.findall(r'[a-z0-9]+', term.lower()))
    rhs = set(re.findall(r'[a-z0-9]+', match.lower()))
    if not lhs or not rhs:
        return 1

    overlap = len(lhs.intersection(rhs)) / max(len(lhs), len(rhs))
    if overlap >= 0.75:
        return 4
    if overlap >= 0.35:
        return 3
    if overlap >= 0.15:
        return 2
    return 1

def _normalized_tokens_for_similarity(text):
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    normalized = []
    for token in tokens:
        # Lightweight stemming to compare lexical cores.
        if len(token) > 4 and token.endswith('ies'):
            token = token[:-3] + 'y'
        elif len(token) > 4 and token.endswith('es'):
            token = token[:-2]
        elif len(token) > 3 and token.endswith('s'):
            token = token[:-1]
        elif len(token) > 5 and token.endswith('al'):
            token = token[:-2]
        normalized.append(token)
    return set(normalized)

def calibrate_rating(term, match, rating):
    """
    Prevent overconfident high ratings when lexical overlap is weak.
    This acts as a guardrail on top of the LLM rating.
    """
    if rating <= 3:
        return rating

    lhs = _normalized_tokens_for_similarity(term)
    rhs = _normalized_tokens_for_similarity(match)
    if not lhs or not rhs:
        return min(rating, 3)

    overlap = len(lhs.intersection(rhs)) / max(len(lhs), len(rhs))
    if rating >= 5 and overlap < 0.75:
        rating = 4
    if rating >= 4 and overlap < 0.45:
        rating = 3
    return rating

def looks_like_english(text):
    text_lower = text.lower()
    if re.search(r'[áéíóúñ¿¡]', text_lower):
        return False

    english_markers = {
        ' the ', ' and ', ' with ', ' was ', ' were ', ' is ', ' are ',
        ' of ', ' in ', ' to ', ' for ', ' patient ', 'year-old',
    }
    hits = sum(1 for marker in english_markers if marker in f' {text_lower} ')
    return hits >= 2

def identify_language(lines):
    # Join lines together (up to first 12) in case certain lines are headings or comments
    # that are too short to determine the language from.
    text = ' // '.join(lines[:12])
    # Use a longer extract to improve reliability.
    text_extract = text[:240].strip()
    if not text_extract:
        return 'english'

    response = create_chat_completion(from_prompt(language_prompts, text_extract), max_tokens=16).strip().lower()
    if response:
        return response
    if looks_like_english(text_extract):
        return 'english'
    return 'unknown'

def as_english(text, language_description):
    """Return the text translated to English (if required)."""
    # If the language isn't English, translate the text depending on the language detected.
    response = language_description
    if 'en' in response or 'english' in response:
        pass
    elif 'es' in response or 'spanish' in response:
        text = create_chat_completion(from_prompt(translate_es_en_prompts, text))
        print(f'Language: {response} ({COLOR_BLUE}translating es -> en{COLOR_RESET})')
    elif looks_like_english(text):
        print(f'Language: {response} ({COLOR_BLUE}fallback detected english; skipping translation{COLOR_RESET})')
    else:
        text = create_chat_completion(from_prompt(translate_en_prompts, text))
        print(f'Language: {response} (unexpected response; {COLOR_BLUE}translating any language -> en{COLOR_RESET})')
    return text

def from_prompt(prompts, term):
    prompts[-1]['content'] = term
    return prompts

def normalize_extracted_entity_text(term_text):
    text = clean_string(term_text)
    # Remove trailing duration/time qualifiers while keeping the clinical concept.
    text = re.sub(
        r"\s+(?:of|for)\s+\d+\s*(?:day|week|month|year|hour|minute)s?(?:['’]?\s*duration)?\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+since\s+(?:yesterday|today|last\s+\w+)\.?$", "", text, flags=re.IGNORECASE)
    return clean_string(text.rstrip(".,;:"))

def identify(text):
    """Return the clinical entities in a clinical note or sample of free text."""
    progress_print('Identifying entities')
    response_json = create_structured_response(
        from_prompt(extract_prompts, text),
        schema=ENTITY_EXTRACTION_SCHEMA,
        schema_name="clinical_entities",
        max_tokens=512,
        use_cache=False,
    )
    # Accept both the new schema {"entities": [...]} and the legacy top-level array [...].
    if isinstance(response_json, dict):
        results = response_json.get("entities")
    elif isinstance(response_json, list):
        results = response_json
    else:
        return {}
    if not isinstance(results, list):
        return {}

    response_terms = []
    for result in results:
        if not isinstance(result, dict):
            continue
        term_text = result.get('text')
        if isinstance(term_text, str):
            term_text = normalize_extracted_entity_text(term_text)
            if term_text:
                response_terms.append(term_text)

    # Dictionary to assign each term a list of information about the match found, confidence and strategy used.
    term_results = {}

    for term in response_terms:
        term_results[term] = ['', 0, NO_MATCH, '', '', '']
        progress_print(f'Matching: {term}')

        # Look for direct matches with SNOMED CT
        potential_match = match_snomed(term)
        if potential_match:
            rating = rate(term, potential_match['display'], text)
            term_results[term] = [potential_match, rating, DIRECT_MATCH, '', '', '']
            if rating > 3:
                continue
        else:
            # Always replace
            rating = 0

        progress_print(f'Prefix searching: {term}')
        prefix_match, prefix_used = match_snomed_by_prefix(term, context=term)
        if prefix_used:
            term_results[term][3] = prefix_used
        if prefix_match is not None and prefix_match != potential_match:
            prefix_rating = rate(term, prefix_match['display'], text)
            if prefix_rating > rating:
                potential_match, rating = prefix_match, prefix_rating
                term_results[term] = [potential_match, rating, PREFIX_MATCH, prefix_used or '', '', '']
            if prefix_rating > 3:
                continue

        progress_print(f'Simplifying: {term}')
        # Attempt to simplify the term in order to improve on the initial match or find a match
        simple_term = create_chat_completion(from_prompt(simplify_prompts, term), max_tokens=16)
        # ignore text after a new line
        simple_term = simple_term.split('\n')[0]
        # clean the string
        simple_term = clean_string(simple_term)
        if not simple_term or simple_term.lower() == term.lower():
            fallback_simple = heuristic_simplify_term(term)
            if fallback_simple and fallback_simple.lower() != term.lower():
                simple_term = fallback_simple

        term_results[term][4] = simple_term

        new_potential_match = match_snomed(simple_term, context=term)
        # Don't repeat queries to the LLM (implicitly: skip if both entries are None)
        if new_potential_match is not None and new_potential_match != potential_match:
            new_rating = rate(simple_term, new_potential_match['display'], term)
            if new_rating > rating:
                # Set the match or replace the previous match with the new match
                potential_match, rating = new_potential_match, new_rating
                term_results[term] = [potential_match, rating, SIMPLIFIED_MATCH, term_results[term][3], simple_term, term_results[term][5]]
            if new_rating > 3:
                continue

        progress_print(f'Generalizing: {term}')
        # Attempt to generalise the term
        general_term = create_chat_completion(from_prompt(generalise_prompts, term), max_tokens=16)
        general_term = clean_string(general_term.split('\n')[0])
        if not general_term:
            general_term = term
        term_results[term][5] = general_term

        # We search using the generalised term but still pick the best match based on our original term
        new_potential_match = match_snomed(general_term, context=term)
        if new_potential_match is not None and new_potential_match != potential_match:
            new_rating = rate(general_term, new_potential_match['display'], term)
            if new_rating > rating:
                # Set the match or replace the previous match with the new match
                potential_match, rating = new_potential_match, new_rating
                term_results[term] = [new_potential_match, new_rating, GENERALISED_MATCH, term_results[term][3], term_results[term][4], general_term]
            # We use a lower threshold, since later steps should only be used if our current match is poor
            if new_rating >= 3:
                continue

    return term_results

def clean_string(s):
    # Remove trailing and leading whitespace
    s = s.strip()
    # Replace multiple newlines with a single newline
    s = re.sub('\n+', '\n', s)
    # Replace multiple spaces with a single space
    s = re.sub(' +', ' ', s)
    return s

def main():
    # Initialise the LLM we are using (if required)
    # Read the test cases (hide blank lines)
    # check if sentences argument is set
    if _args.sentences:
        with open(_args.sentences, "r") as file:
            stripped_lines = map(str.rstrip, file.readlines())
        # skip newlines and comments/titles    
        lines = [line for line in stripped_lines if line and not line.startswith('#')]
        if _args.n is not None:
            if _args.n < 1:
                raise ValueError('--n must be a positive integer.')
            lines = lines[:_args.n]
        
        start_time_all_cases = time.time()
        entities_per_line = []
        total_ok = 0
        total_entities = 0
        language = identify_language(lines)

        # Iterate over each line in the test cases
        for index, line in enumerate(lines, start=1):
            if output_mode == 'summary':
                print(f'Processing sentence {index}...')
            else:
                print(COLOR_BLUE, line, COLOR_RESET, sep='')
            start_time = time.time()
            text = as_english(line, language)  # translate text in other languages to english
            entities = identify(text)  # identify entities
            entities_per_line.append(entities)
            line_ok, line_total = sentence_score_counts(entities, min_rating=OK_MIN_RATING)
            total_ok += line_ok
            total_entities += line_total
            elapsed = time.time() - start_time
            if output_mode == 'summary':
                display_summary(text, entities, elapsed)
            else:
                display_color(text, entities, elapsed)

        total_time = time.time() - start_time_all_cases
        total_score_pct = round((total_ok / total_entities) * 100, 1) if total_entities else 0.0
        if output_mode == 'summary':
            print(COLOR_BLUE, f'Total time: {round(total_time, 1)}s | Overall score (green+yellow/total): {total_ok}/{total_entities} ({total_score_pct}%)', COLOR_RESET, sep='')
        else:
            print(COLOR_BLUE, f'Total time: {round(total_time, 1)}s | Overall score (OK={OK_MIN_RATING}-5): {total_ok}/{total_entities} ({total_score_pct}%)', COLOR_RESET, sep='')
    else:
        raise ValueError('Please provide a path to a set of free text clinical sentences to encode with SNOMED CT.')
    
if __name__ == "__main__":
    main()
