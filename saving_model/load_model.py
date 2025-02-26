import joblib

model = joblib.load('dib_79.pkl')

output = model.predict([[0,0,0,0,0,0,0,0]])

print(output)

if output[0] == 1:
    print('diabatic')
else:
    print('not diabatic')

    