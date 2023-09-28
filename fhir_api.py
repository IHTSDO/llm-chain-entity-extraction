import subprocess
import json

def expand_valueset(server_url, valueset_url, filter_value=None, fuzzy_match=True):
    headers = {
        'User-Agent': 'Implementation Team AI testing (Llama LLM)',
        'Accept': 'application/json'  # Ensure we get JSON response
    }

    # Define the operation endpoint
    endpoint = "/ValueSet/$expand"

    # Define the request URL
    url = server_url + endpoint

    # Define the query parameters
    params = {
        "url": valueset_url,
        "count": "5",
        "offset": "0",
        "displayLanguage": "en",
        "language": "en"
    }

    if fuzzy_match:
        filter_value = filter_value + "~"

    # Add filter parameter if provided
    if filter_value:
        params["filter"] = filter_value

    # Construct the curl command
    command = ["curl", "-s"]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])

    command.append("-G")
    command.append(url)

    for key, value in params.items():
        command.extend(["--data-urlencode", f"{key}={value}"])

    # Execute the curl command and get the output
    completed_process = subprocess.run(command, capture_output=True, text=True)
    
    # Check the response
    if completed_process.returncode == 0:
        try:
            expanded_valueset = json.loads(completed_process.stdout)
            return expanded_valueset
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON response.")
    else:
        print("Error with curl command execution.")
        print("Return code:", completed_process.returncode)
        print("Error details:", completed_process.stderr)
        print("Command executed:", " ".join(command))

#     # Send the request
#     response = requests.get(url, params=params, headers=headers)

#     # Check the response status
#     if response.status_code == 200:
#         # Extract the expanded ValueSet from the response
#         expanded_valueset = response.json()

#         # Return the expanded ValueSet
#         return expanded_valueset
#     else:
#         print("Error:", response.status_code)
#         print(response.text)