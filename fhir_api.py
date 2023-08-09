import requests

def expand_valueset(server_url, valueset_url, filter_value=None):
    headers = {'User-Agent': 'Implementation Team AI testing (Llama LLM)'}

    # Define the operation endpoint
    endpoint = "/ValueSet/$expand"

    # Define the request URL
    url = server_url + endpoint

    # Define the query parameters
    params = {
        "url": valueset_url,
        "count": 5,
        "offset": 0,
        "displayLanguage": "en",
        "language": "en"
    }

    # Add filter parameter if provided
    if filter_value:
        params["filter"] = filter_value

    # Send the request
    response = requests.get(url, params=params, headers=headers)

    # Check the response status
    if response.status_code == 200:
        # Extract the expanded ValueSet from the response
        expanded_valueset = response.json()

        # Return the expanded ValueSet
        return expanded_valueset
    else:
        print("Error:", response.status_code)
        print(response.text)

# Example usage
# server_url = "https://snowstorm.ihtsdotools.org/fhir"
# valueset_url = "http://snomed.info/sct/900000000000207008/version/20230630?fhir_vs"
# filter_value = "asthma"

# expanded_valueset = expand_valueset(server_url, valueset_url, filter_value)
# print(expanded_valueset)
# Process the expanded ValueSet as needed
# ...
