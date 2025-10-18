# import numpy as np
# import pickle
# import pandas as pd
# import os
# from flask import Flask, request, jsonify, render_template

# app = Flask(__name__)

# # Load trained model and scaler
# model = pickle.load(open('model.pkl', 'rb'))
# scale = pickle.load(open('encoder.pkl', 'rb'))  # Ensure this was trained with correct feature names

# @app.route('/')  # Home page
# def home():
#     return render_template("index.html")  # Rendering the home page


# @app.route('/predict', methods=["POST", "GET"])  # Prediction route
# def predict():
#     try:
#         # Read input values and ensure all expected features are present
#         input_feature = [float(x) for x in request.form.values()]
        
#         # Define correct feature order
#         categorical_columns = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day','hours', 'minutes', 'seconds']

#         # Ensure the number of input features matches expected features
#         if len(input_feature) != len(categorical_columns):
#             return jsonify({"error": "Feature mismatch. Expected {}, but got {}.".format(len(), len(input_feature))})
        
#         # Convert to DataFrame
#         data = pd.DataFrame([input_feature], columns=categorical_columns)
#         print("Feature names in input data:", data.columns)  # Debugging step

#         # Transform input data
#         data = scale.transform(data)

#         # Predict using the trained model
#         prediction = model.predict(data)

#         # Return the result
#         return render_template("ouput.html", prediction_text="Estimated Traffic Volume: " + str(prediction[0]))

#     except Exception as e:
#         return jsonify({"error": str(e)})


# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 5000))
#     app.run(port=port, debug=True, use_reloader=False)




import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('encoder.pkl', 'rb'))

# Define final column order from training
column_order = [
    'temp', 'rain', 'snow', 'day', 'month', 'year',
    'weather_Clear', 'weather_Clouds', 'weather_Drizzle', 'weather_Fog',
    'weather_Haze', 'weather_Mist', 'weather_Rain', 'weather_Smoke',
    'weather_Snow', 'weather_Squall', 'weather_Thunderstorm'
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Step 1: Get form data
        form_data = request.form.to_dict()

        # Step 2: Extract numeric features
        numeric_data = {
            'temp': float(form_data['temp']),
            'rain': float(form_data['rain']),
            'snow': float(form_data['snow']),
            'day': float(form_data['day']),
            'month': float(form_data['month']),
            'year': float(form_data['year'])
        }

        # Step 3: One-hot encode weather
        weather_value = form_data['weather']
        weather_encoded = {col: 0 for col in column_order if col.startswith('weather_')}
        weather_key = f'weather_{weather_value}'
        if weather_key in weather_encoded:
            weather_encoded[weather_key] = 1

        # Step 4: Combine features
        final_input = {**numeric_data, **weather_encoded}
        input_df = pd.DataFrame([final_input])

        # Step 5: Reorder columns to match training
        input_df = input_df.reindex(columns=column_order, fill_value=0)

        # Step 6: Scale input
        input_scaled = scaler.transform(input_df)

        # Step 7: Predict
        prediction = model.predict(input_scaled)

        # Step 8: Return result
        return render_template("output.html", prediction_text=f"Estimated Traffic Volume: {int(prediction[0])}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)