import requests
import json

url = "https://vchekservicedemoapp.azurewebsites.net/v1/PersonAnalytics"

payload = json.dumps({
  "gender": "Male",
  "isBagAvailable": 0,
  "isPersonIn": 1,
  "detectedDateTime": "2023-10-04 15:30:20"
})
headers = {
  'apikey': '415d1f410a424a4ba0e6925991db57b2',
  'Content-Type': 'application/json',
  'Cookie': 'ARRAffinity=79e06db539acb57119e709978d2cf1da299e8341753d6f6345007fcab3f69bc5; ARRAffinitySameSite=79e06db539acb57119e709978d2cf1da299e8341753d6f6345007fcab3f69bc5'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
