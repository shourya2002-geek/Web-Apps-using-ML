import requests

url = "http://localhost:5000/predict_api"
r = request.post(url,json{'expereince':2, 'test_score':9, 'interview_score':9})

print(r.json())