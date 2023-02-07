import requests

data = {'age': 33,
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
aws = "http://mlops-loadb-1vls10v8z56rd-8e8763d7d81b9439.elb.us-east-1.amazonaws.com:8000"

r = requests.post(f'{aws}/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
