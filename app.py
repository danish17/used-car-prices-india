from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, static_url_path="/static")

#Max Age Zeroing to serve fresh
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def pipeline(features):
    name_enc = pickle.load(open("models/encoders/name.enc", "rb"))
    loc_enc = pickle.load(open("models/encoders/location.enc", "rb"))
    fuel_enc = pickle.load(open("models/encoders/fuel.enc", "rb"))
    trans_enc = pickle.load(open("models/encoders/trans.enc", "rb"))
    owner_enc = pickle.load(open("models/encoders/owner.enc", "rb"))
    km_scaler = pickle.load(open("models/scale.scl", "rb"))

    make = features[0]
    location = features[1]
    year = features[2]
    km = features[3]
    fuel = features[4]
    trans = features[5]
    owner = features[6]
    mileage = features[7]
    engine = features[8]
    power = features[9]
    seats = features[10]

    make = make.lower()
    location = location.lower()

    make = name_enc.transform(np.array([make]).reshape(-1,1))
    location = loc_enc.transform(np.array([location]).reshape(-1,1))
    fuel = fuel_enc.transform(np.array([fuel]).reshape(-1,1))
    trans = trans_enc.transform(np.array([trans]).reshape(-1,1))
    owner = owner_enc.transform(np.array([owner]).reshape(-1,1))
    km = km_scaler.transform(np.array([km]).reshape(-1,1))

    return [make,location,year,km,fuel,trans,owner,mileage,engine,power,seats]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/404')
def error():
    return render_template('404.html')

@app.route('/form')
def feedback():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    make = request.form.get('make')
    location = request.form.get('location')
    year = request.form.get('year')
    km = request.form.get('km')
    fuel = request.form.get('fuel')
    trans = request.form.get('trans')
    owner = request.form.get('owner_type')
    mileage = request.form.get('mile')
    engine = request.form.get('engine')
    power = request.form.get('power')
    seats = request.form.get('seats')

    to_pass = [make,location,year,km,fuel,trans,owner,mileage,engine,power,seats]

    processed = pipeline(to_pass)

    model = pickle.load(open("models/rf.mdl","rb"))

    price = model.predict(np.array(processed).reshape(1,-1))[0]

    message = "Predicted price: {} lacs".format(price)

    return render_template('index.html',
                           price = message)

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
