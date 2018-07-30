import requests
r = requests.post('http://65.19.181.36:23100/infer', data = {'data':'base64img'})
print(r.text)
