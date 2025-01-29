import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "n": 10,
    "p": 50,
    "k": 20,
    "temperature": 25.0,
    "humidity": 70.0,
    "ph": 6.5,
    "rainfall": 100.0
}

response = requests.post(url, json=data)
print(response.json())
