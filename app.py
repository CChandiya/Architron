from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
from dotenv import load_dotenv
from datetime import datetime, timedelta
import secrets
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))
app.config['SESSION_COOKIE_SECURE'] = os.getenv('ENVIRONMENT', 'development') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Connect to MongoDB
try:
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(mongo_uri)
    db = client[os.getenv('MONGO_DB', 'user_auth_db')]
    users_collection = db['users']
    # Ensure indices
    users_collection.create_index('username', unique=True)
    users_collection.create_index('email', unique=True)
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    raise

# Streamlit app configuration
STREAMLIT_URL = os.getenv('STREAMLIT_URL', 'http://localhost:8501')

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def validate_password(password):
    """Validates password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password) or not re.search(r'[a-z]', password):
        return False, "Password must contain both uppercase and lowercase letters"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"

def validate_email(email):
    """Validates email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    return False

# Routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            email = request.form.get('email')
            
            # Input validation
            if not username or not password or not email:
                return render_template('register.html', error='All fields are required')
            
            if not validate_email(email):
                return render_template('register.html', error='Invalid email format')
            
            is_valid, message = validate_password(password)
            if not is_valid:
                return render_template('register.html', error=message)
            
            # Check if username or email already exists
            if users_collection.find_one({'username': username}):
                return render_template('register.html', error='Username already exists')
            
            if users_collection.find_one({'email': email}):
                return render_template('register.html', error='Email already exists')
            
            # Hash the password
            hashed_password = generate_password_hash(password)
            
            # Insert new user
            user_id = users_collection.insert_one({
                'username': username,
                'email': email,
                'password': hashed_password,
                'created_at': datetime.utcnow(),
                'last_login': None
            }).inserted_id
            
            flash('Account created successfully! Please log in.', 'success')
            logger.info(f"New user registered: {username}")
            return redirect(url_for('home'))
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return render_template('register.html', error='An error occurred. Please try again later.')
    
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        # Input validation
        if not username or not password or not email:
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if not validate_email(email):
            return jsonify({'success': False, 'message': 'Invalid email format'})
        
        is_valid, message = validate_password(password)
        if not is_valid:
            return jsonify({'success': False, 'message': message})
        
        # Check if username or email already exists
        if users_collection.find_one({'username': username}):
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        if users_collection.find_one({'email': email}):
            return jsonify({'success': False, 'message': 'Email already exists'})
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Insert new user
        user_id = users_collection.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.utcnow(),
            'last_login': None
        }).inserted_id
        
        logger.info(f"API: New user registered: {username}")
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except Exception as e:
        logger.error(f"API registration error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred. Please try again later.'})

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        remember = data.get('remember', False)
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'})
        
        # Find the user
        user = users_collection.find_one({'username': username})
        
        if not user or not check_password_hash(user['password'], password):
            logger.warning(f"Failed login attempt for username: {username}")
            return jsonify({'success': False, 'message': 'Invalid username or password'})
        
        # Update last login time
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Set session
        session['user_id'] = str(user['_id'])
        session['username'] = user['username']
        
        if remember:
            # If remember me is checked, set permanent session
            session.permanent = True
        
        logger.info(f"User logged in: {username}")
        return jsonify({'success': True, 'message': 'Login successful'})
        
    except Exception as e:
        logger.error(f"API login error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred. Please try again later.'})

@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html', error='Username and password are required')
        
        # Find the user
        user = users_collection.find_one({'username': username})
        
        if not user or not check_password_hash(user['password'], password):
            logger.warning(f"Failed login attempt for username: {username}")
            flash('Invalid username or password', 'error')
            return render_template('login.html', error='Invalid username or password')
        
        # Update last login time
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Set session
        session['user_id'] = str(user['_id'])
        session['username'] = user['username']
        
        if remember:
            # If remember me is checked, set permanent session
            session.permanent = True
        
        logger.info(f"User logged in: {username}")
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        flash('An error occurred. Please try again later.', 'error')
        return render_template('login.html', error='An error occurred. Please try again later.')

@app.route('/logout')
def logout():
    # Remove user session
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect to Streamlit application
    streamlit_url = STREAMLIT_URL
    
    # You can add query parameters to pass user info to Streamlit if needed
    # streamlit_url += f"?username={session['username']}"
    
    logger.info(f"User {session['username']} redirected to Streamlit app")
    return redirect(streamlit_url)

@app.route('/forgot-password')
def forgot_password():
    # In a real application, implement password reset functionality
    flash('Password reset functionality is not implemented in this demo', 'info')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.getenv('ENVIRONMENT', 'development') == 'development', host='0.0.0.0', port=port)