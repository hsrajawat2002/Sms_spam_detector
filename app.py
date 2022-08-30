from flask import Flask,render_template,url_for,request
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as tf
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))
#ex=pickle.load(open('model2.pkl','rb'))
ex=tf(min_df=1,stop_words='english',lowercase='True')
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = ex.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)