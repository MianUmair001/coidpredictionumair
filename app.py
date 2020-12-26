from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("covid19.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    print(int_features)
    final={'Yesterday':int_features[0],'Yesterday_diff':int_features[1]}
    final=pd.DataFrame(final,index=[0])
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output='{0:.{1}f}'.format(prediction[0], 2)
    return render_template('covid19.html',pred='Probability of New Cases is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
