from bardapi import Bard, BardCookies
from prompts import convert_chat_to_message


with open('bard.key', 'r') as file:
   token_1PSID, token_1PSIDTS, token_1PSIDCC, *_other_lines = map(str.strip, file.readlines())

cookie_dict = {
    "__Secure-1PSID": token_1PSID,
    "__Secure-1PSIDTS": token_1PSIDTS,
    "__Secure-1PSIDCC": token_1PSIDCC
}
print(cookie_dict, end='\r')

#bard = BardCookies(cookie_dict=cookie_dict)
bard = Bard(token=token_1PSID)

def sanitise_bard_message(raw_response, prefer_length=4000):
    # Return only the most response-like line that falls under the preferred character count.
    best_line = ''
    for line in raw_response.split('\n'):
        # Ignore newlines and remove Bard's long-winded explanatory text
        if line and not line.startswith('I understand') and not line.startswith('Here are ') and not line.startswith('Sure ') and not (line.startswith('I ') and line.endswith('.')):
            if best_line == '' or (len(best_line) > prefer_length and len(best_line) > len(line)):
                best_line = line
    # Bard often creates bullet point lists with * and writes quotes or code blocks with ```
    return best_line.strip('`* \n')

def create_chat_completion(prompts, max_tokens=4000, **_):
    # We ignore keyword arguments as the Bard "unofficial API" cannot be configured.
    # Bard only takes strings, so we convert the prompt into a single string
    message = convert_chat_to_message(prompts)
    print('Message:', message)
    raw_response = bard.get_answer(message)['content']
    return sanitise_bard_message(raw_response, prefer_length=max_tokens * 2)
