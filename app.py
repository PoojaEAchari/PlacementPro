from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load the trained model
model = pickle.load(open('logistic_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve values from the request
        stream = float(request.args.get('stream', 0))  # Default to 0 if not provided
        cgpa = float(request.args.get('cgpa', 0))    # Default to 0 if not provided
        backlogs = float(request.args.get('backlogs', 0))  # Default to 0 if not provided
        
        # Prepare input for the model
        arr = np.array([backlogs, stream, cgpa]).reshape(1, -1)  # Reshape for single sample
        output = model.predict(arr)
        
        # Interpret the output
        if output[0] == 1:
            result = 'Eligible'
        else:
            result = 'Not Eligible'
        
        return jsonify({'result': result})
    
    except Exception as e:
        # Handle errors and provide a default message
        return jsonify({'result': f'Error occurred: {str(e)}'})

@app.route('/eligible_students')
def eligible_students():
    try:
        # Load the dataset directly within this route
        dataset_path = 'PLACEMENTPRO.csv'  # Path to your dataset file
        df = pd.read_csv(dataset_path)
        
        # Filter eligible students
        eligible_students = df[
            (df['Backlogs'] == 0) & (df['CGPA'] > 7)
        ]
        
        # Convert the DataFrame to a list of dictionaries
        students_data = eligible_students[['Name', 'SRN', 'CGPA', 'Backlogs', 'Stream', 'Placed/not']].to_dict(orient='records')
        
        return render_template('students_table.html', students=students_data)
    
    except Exception as e:
        # Handle errors and provide a default message
        return f'Error occurred: {str(e)}'

if __name__ == "__main__":
    app.run(debug=True)
