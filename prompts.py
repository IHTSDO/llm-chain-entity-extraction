
# Define prompt formats
extract_prompts_listed = [ { "role": "system", "content": """You are a clinical entity extractor. \
Review the clinical notes provided by the user and extract the important entities mentioned. \
First extract all symptoms, then extract all diagnoses, then extract all procedures, and finally extract all medications. \
Each clinical note is independent. Don't include demographics, commentary or clarification, only the entities and the requested properties. \

The desired format is one comma-separated list for each type of entity.
Symptoms: <comma_separated_list_of_symptoms>
Diagnoses: ...
Procedures: ...
Medications: ...
"""},
    {"role":"user", "content":"A 76-year-old man with a history of chronic back pain presented with dizziness and altered mental status. A colonoscopy evidenced ulcerative colitis."},
    {"role":"assistant", "content":'[{"text":"chronic back pain"}, {"text":"dizziness"}, {"text":"altered mental status"}, {"text":"anion-gap metabolic acidosis"}, {"text":"colonoscopy"}, {"text":"ulcerative colitits"}]'},
    {"role":"user", "content":""} ]

extract_prompts = [
    { "role": "system", "content": """You are a clinical entity extractor. Report results as a JSON array of objects. \
Review the clinical notes provided by the user and extract all mentions of symptoms, diagnoses, procedures, and medications. \
Each clinical note is independent. \
Don't include demographics, commentary or clarification, only entities and the requested properties. \
Provide the following information for each entity:
 - text: text of the entity
 - type: type of clinical entity (symptom, diagnosis, procedure, medication, etc.)
 - context: present or absent"""},
    {"role":"user", "content":""}
    ]


simplify_prompts = [ { "role": "system", "content": """You are a clinical entity simplifier. Respond with simpler forms of the terms provided by the user. \
The goal is to make the terms easier to match with SNOMED.
SNOMED is a clinical terminology that does not use plurals or other non-essential words. Remove plurals, and other non-essential words.
Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use.
"""},
    {"role":"user", "content":"pain in hands"},
    {"role":"assistant", "content":"hand pain"},
    {"role":"user", "content":"multiple vesicular lesions"},
    {"role":"assistant", "content":"vesicular lesion"},
    {"role":"user", "content":"severe allergic symptoms"},
    {"role":"assistant", "content":"allergic disorder"},
    {"role":"user", "content":""} ]

'''generalise_prompts_LEGACY = [ { "role": "system", "content": """Your task is to answer in a consistent style.
Provide a new short and concise medical term without the specificity of the original term.
The goal is to make the terms easier to match with SNOMED.
SNOMED is a clinical terminology that does not use plurals or other non-essential words. Remove plurals, and other non-essential words.
Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use.
"""},
    {"role":"user", "content":"large pleural effusion"},
    {"role":"assistant", "content":"pleural effusion"},
    {"role":"user", "content":"intermittent asthma"},
    {"role":"assistant", "content":"asthma"},
    {"role":"user", "content":""} ]'''

generalise_prompts = [ { "role": "system", "content": """Given a specific clinical term, provide a broader category that encompasses the term. \
The goal is to make the terms easier to match with SNOMED.
SNOMED is a clinical terminology that does not use plurals or other non-essential words. Remove plurals, and other non-essential words.
Do not include any commentary or explanation in your response. Only a clinical term like a clinician would use."""},
    {"role":"user", "content":"vision loss"},
    {"role":"assistant", "content":"visual impairment"},
    {"role":"user", "content":"asthma"},
    {"role":"assistant", "content":"respiratory finding"},
    {"role":"user", "content":"secondary hypertension"},
    {"role":"assistant", "content":"hypertension"},
    {"role":"user", "content":""} ]

'''accuracy_prompts_LEGACY = [ {"role":"system", "content": """You are a clinical expert, that compares terms doctors write in clinical notes with SNOMED CT terms selected to represent the same meaning.
You will be given two clinical terms and you need to assess how closely they are related on a scale from 1 to 5, where 1 means no meaningful relationship and 5 means identical meaning.
Do not include any commentary or explanation in your response.
"""}, 
    {"role":"user", "content":""}]'''

accuracy_prompts = [ {"role":"system", "content": """You are a clinical expert, that compares terms doctors write in clinical notes with SNOMED CT terms selected to represent the same meaning.
You will be given a clinician's term and a SNOMED term, and you need to assess how closely the terms are related based on the given context, on a scale from 1 to 5, where 1 means no meaningful relationship and 5 means identical meaning.
Do not include any commentary or explanation in your response. Use this table for guidance:

1 - No meaningful relationship
2 - Loosely related
3 - Related but not synonymous
4 - Synonymous
5 - Identical meaning
"""}, 
    {"role":"user", "content":""}]

# LLaMA refuses to only output a number, so we allow it to 'briefly' explain its thoughts.
# LLaMA is far more confident, so the decriptions of 3-5 have been shifted.
# Without the additional specificity, it also confuses greater detail as being a good thing.
accuracy_prompts_llama = [ {"role":"system", "content": """You are a clinical expert, that compares terms doctors write in clinical notes with SNOMED CT terms selected to represent the same meaning.
You will be given a correct clinician's term, a SNOMED term, and their context. Briefly assess how closely related and similar in specificity the terms are on a scale from 1 to 5, following this table:

1 - No meaningful relationship
2 - Loosely related
3 - Synonymous
4 - Nearly identical and equally specific
5 - Exactly identical and equally specific
"""}, 
    {"role":"user", "content":""}]

select_best_prompts = [ {"role":"system", "content": """You are a clinical expert, that compares terms doctors write in clinical notes with SNOMED terms selected to represent the same meaning.
You will be given a clinical term and a list of possible matching terms from SNOMED. You need to select the single best match from the list of possible matching terms.
Do not include any commentary or explanation in your response, only the term which is the best match.
"""}, 
    {"role":"user", "content":""}]

#swap_spelling_prompts = [ {"role":"system", "content": """You will given a medical term and its context. Respell the provided term, by first swapping the spelling between American English and British English, then removing complex adjectives.
#Do not include any commentary or explanation in your response, only the new spelling of the new term.
#"""}, 
#    {"role":"user", "content":""}]

language_prompts = [ {"role":"system", "content": """Tell me the language in which this extract from a clinical note is written, without translating the text.
Do not include any commentary or explanation in your response, only the name of the language.
"""},
    {"role":"user", "content":""}]

translate_es_en_prompts = [ {"role":"system", "content": """Translate this text from Spanish to English.
Do not include any commentary or explanation in your response.
"""},
    {"role":"user", "content":""}]

translate_en_prompts = [ {"role":"system", "content": """Translate this text to English.
Do not include any commentary or explanation in your response.
"""},
    {"role":"user", "content":""}]


def convert_chat_to_message(prompts):
    if len(prompts) == 2:  # Zero-shot
        message = f'{prompts[0]["content"]}\nText: """\n{prompts[1]["content"]}\n"""'
    else:  # Few-shot
        lines = [prompts[0]['content']]
        lines.extend(f"{prompt['role'].capitalize()}: {prompt['content']}" for prompt in prompts[1:])
        message = '\n'.join(lines)
    return message
