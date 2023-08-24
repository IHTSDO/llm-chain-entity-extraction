"""Extracts clinical entities from free text using LLMs."""

from entity_extractor import *

def disp(line, ann_line, entities, time_taken):
    ann = [a.strip() for a in ann_line.split('\n')]

    print(colorize_text(line, entities.keys(), COLOR_GREEN))
    print(f'(overlapping matches not highlighted)\n{len(entities.keys())} entities found in {round(time_taken, 1)}s:')
    print(ann)
    for key, value in entities.items():
        if value is None or value[2] == NO_MATCH:
            match, rating, *method_info = value
            print(f'{key} {COLOR_RED}No match{COLOR_RESET} {method_info}')
            continue
        
        match, rating, *method_info = value
        # color = {5: COLOR_GREEN, 4: COLOR_YELLOW}.get(rating, COLOR_RED)

        if rating < 3:
            color = COLOR_BLUE
        elif any(match['code'] in b for b in ann):
            color = COLOR_GREEN
        elif any(match['display'] in b for b in ann):
            color = COLOR_YELLOW
        else:
            color = COLOR_RED

        print(f'{color}{key}{COLOR_RESET}: {COLOR_BLUE}{match["display"]} |{match["code"]}| {color}(confidence {rating}){COLOR_RESET} {method_info}')
    
    # Fail case where entities = {}
    if not entities:
        print(f'{COLOR_RED}Could not identify.{COLOR_RESET}')

    print(COLOR_BLUE, '---', COLOR_RESET)

def main():
    # Initialise the LLM we are using (if required)
    # Read the test cases (hide blank lines)
    with open("clinical_text_3.txt", "r") as file:
        stripped_lines = map(str.rstrip, file.readlines())

    # skip newlines and comments/titles    
    lines = [line for line in stripped_lines if line and not line.startswith('#')]
    
    with open("clinical_text_3_ann.txt", "r") as file:
        ann_lines = file.read().split(';;')
    
    start_time_all_cases = time.time()
    entities_per_line = []
    language = identify_language(lines)

    # Iterate over each line in the test cases
    for ann_line, line in zip(ann_lines, lines):
        start_time = time.time()
        text = as_english(line, language)  # translate text in other languages to english
        entities = identify(text)  # identify entities
        entities_per_line.append(entities)
        disp(text, ann_line, entities, time.time() - start_time)

    print(COLOR_BLUE, time.time() - start_time_all_cases, 's', COLOR_RESET, sep='')

if __name__ == "__main__":
    main()
