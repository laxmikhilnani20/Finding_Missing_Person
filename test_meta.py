import requests
import re
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
text = requests.get('https://www.instyle.com/anne-hathaway-oscars-2026-returns-after-12-years-11920535', headers=headers).text
match = re.search(r'property="og:image"\s+content="([^"]+)"', text)
if match:
    print("Found:", match.group(1))
