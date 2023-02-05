import requests

data = {'age': 19,
        'workclass': 'Private',
        'fnlgt': 149184,
        'education': 'HS-grad',
        'marital_status': 'Never-married',
        'occupation': 'Prof-specialty',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'hoursPerWeek': 60,
        'nativeCountry': 'United-States'
        }

# local = "http://127.0.0.1:8000"
aws = "http://mlops-loadb-15esodbag9ua0-f444e892dfcef8b8.elb.us-east-1.amazonaws.com:8000"

r = requests.post(f'{aws}/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
