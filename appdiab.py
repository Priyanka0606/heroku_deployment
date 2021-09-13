from flask import Flask, render_template, request
import joblib

# initialise the app
app = Flask(__name__)
model = joblib.load('diab_79.pkl')
print('[INFO] model loaded')


@app.route('/')
def hello_world():
    return render_template('diab.html')

@app.route('/predict' , methods = ['post'])
def predict():
    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age = request.form.get('Age')

    print(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin , BMI, DiabetesPedigreeFunction , Age)
    output = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin , BMI, DiabetesPedigreeFunction , Age]])

    if output[0] == 0:
        ans = 'Not Diabetic'
    else:
        ans = 'Diabetic'   
    

    return  render_template('predict.html', predict = f'The person is {ans}')



#run the app
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
