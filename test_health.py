import requests

try:
    r = requests.get('http://localhost:8000/health', timeout=10)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text}")
except Exception as e:
    print(f"Error: {e}")
