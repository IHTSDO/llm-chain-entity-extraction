import requests
import time
import subprocess
import logging
import http.client

# Enable verbose logging for requests
http.client.HTTPConnection.debuglevel = 1
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.3'
}

url = "https://snowstorm-lite.nw.r.appspot.com/fhir/ValueSet/$expand?url=http%3A%2F%2Fsnomed.info%2Fsct%3Ffhir_vs%3Disa%2F138875005&count=5&offset=0&displayLanguage=en&language=en&filter=diplopia~"

# Timing the request using the requests library
start_time_requests = time.time()
response = requests.get(url, allow_redirects=True, headers=headers)
end_time_requests = time.time()
elapsed_time_requests = end_time_requests - start_time_requests
print("Time elapsed using requests:", elapsed_time_requests)

# Timing the request using curl
start_time_curl = time.time()
subprocess.run(["curl", "-s", "-o", "/dev/null", url], capture_output=True)
end_time_curl = time.time()
elapsed_time_curl = end_time_curl - start_time_curl
print("Time elapsed using curl:", elapsed_time_curl)
