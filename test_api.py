import requests

url = "https://api.apilayer.com/sentiment/analysis"
headers = {"apikey": "sW6EbemSUMnsSPYTPkaayOj86yQMcMNh"}
data = {"text": "Test sentiment analysis"}

response = requests.post(url, headers=headers, json=data)
print("Status Code:", response.status_code)
print("Response Body:", response.json())