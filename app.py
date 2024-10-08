from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import sqlite3



# Set up Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['SECRET_KEY'] = 'e2a14f4eecb16a267e5521a7a56b420adfbdbf6f37282995b988de3c1b310f3a'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png'}  # Allow more file types for flexibility


# Load the pre-trained model
print("Loading the pre-trained model...")
model_resnet = load_model("parkinsons_detection_ensemble.hdf5")
print("Model loaded successfully.")

# Function to check allowed file types
def allowed_file(filename):
    print(f"Checking if the file '{filename}' is allowed...")
    result = '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    print(f"File allowed: {result}")
    return result

# Dictionary for label mapping
label_dict = {0: 'Normal', 1: 'Parkinson'}


def create_users_table():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            password TEXT,
            mobile_number TEXT,
            email TEXT UNIQUE
        )
    ''')
    conn.commit()
    conn.close()

def email_exists(email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user is not None

@app.route('/logout')
def logout():
    session.clear()
    return render_template('login.html')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def validate_credentials(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
    user = cursor.fetchone()

    conn.close()

    return user is not None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['username']  # Change 'email' to 'username'
        password = request.form['password']
        
        isValid = validate_credentials(email, password)

        #
        if isValid:
            session['email'] = email
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid email or password')
    else:
        return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']       
        mobile_number = request.form['mobile-number']
        email = request.form['email']
      
        create_users_table()
        
        if email_exists(email):
            return redirect(url_for('signup', email_exists=True))

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password, mobile_number, email) VALUES (?, ?, ?, ?)', (username, password, mobile_number, email))
        conn.commit()
        conn.close()

        return redirect(url_for('login', signup_success=True))
    else:
        return render_template('signup.html')

 
# Function to prepare image for prediction
def prepare_image(filepath, target_size=(224, 224)):
    print(f"Preparing image from: {filepath}")
    img = load_img(filepath, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    img = img / 255.0  # Normalize the image
    print(f"Image prepared with shape: {img.shape}")
    return img

# Route for home page (file upload)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash('No file part. Please upload an image.', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file. Please upload an image.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Secure the filename and save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")

            # Prepare image for prediction
            image = prepare_image(filepath)

            # Perform prediction
            print("Performing prediction...")
            prediction_prob = model_resnet.predict(image)
            predicted_class = np.argmax(prediction_prob, axis=1)[0]
            prediction_label = label_dict[predicted_class]
            print(f"Prediction: {prediction_label} (Probability: {prediction_prob})")

            # Redirect to result page and pass the data
            return redirect(url_for('result', uploaded_image=filename, prediction=prediction_label))

        else:
            flash('Invalid file type. Please upload a PNG file.', 'danger')

    return render_template('upload.html')

# Route for result page
@app.route('/result')
def result():
    # Get the prediction and image from the request arguments
    uploaded_image = request.args.get('uploaded_image')
    prediction = request.args.get('prediction')
    print(f"Displaying result for: {uploaded_image}, Prediction: {prediction}")

    return render_template('result.html', uploaded_image=uploaded_image, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)