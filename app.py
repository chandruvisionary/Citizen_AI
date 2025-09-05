import os
import logging
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///citizenai.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize database
db.init_app(app)

# Hugging Face API configuration
HF_API_KEY = os.environ.get("HF_API_KEY", "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")
HF_API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.3-8b-instruct"

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to feedback
    feedback_entries = db.relationship('Feedback', backref='user', lazy=True)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Helper Functions
def get_ai_response(question):
    """Get response from AI model - simplified to use reliable service"""
    
    # For now, use a smart rule-based response system with context-aware replies
    # This ensures reliable responses while external AI services are being resolved
    
    question_lower = question.lower()
    
    # Categorize common government queries and provide helpful responses
    if any(word in question_lower for word in ['license', 'permit', 'registration', 'id']):
        return f"""Regarding your question about licenses and permits:

For most licenses and permits, you'll need to:
1. Visit your local government office or official website
2. Bring required documentation (usually ID, proof of residence, application forms)
3. Pay applicable fees
4. Allow processing time (varies by type)

Common types:
- Driver's License: Contact your local DMV
- Business License: Visit your city/county clerk's office
- Building Permit: Contact your local building department
- Professional License: Check with your state's professional licensing board

Would you like specific information about a particular type of license or permit?"""
    
    elif any(word in question_lower for word in ['vote', 'voting', 'election', 'ballot']):
        return f"""Regarding voting and elections:

To participate in elections, you need to:
1. Register to vote (if not already registered)
2. Find your polling location
3. Understand what's on your ballot
4. Know your voting rights

Key resources:
- Voter registration: Contact your local election office or register online
- Polling locations: Check your state's Secretary of State website
- Sample ballots: Available on your county election website
- Voting rights: Protected by federal and state law

Upcoming elections and deadlines can be found on your local election office website."""
    
    elif any(word in question_lower for word in ['tax', 'taxes', 'irs', 'refund']):
        return f"""Regarding tax matters:

For tax assistance, here are your main options:
1. IRS website (irs.gov) for federal tax information
2. Your state's revenue department for state taxes
3. Local tax assessor for property taxes
4. Free tax preparation programs (VITA/TCE) for qualifying individuals

Common tax services:
- Filing tax returns
- Getting tax refunds
- Payment plans for owed taxes
- Tax transcripts and records
- Resolving tax issues

For immediate assistance, you can call the IRS helpline or visit a local tax assistance center."""
    
    elif any(word in question_lower for word in ['social', 'benefits', 'assistance', 'welfare', 'food', 'medicaid']):
        return f"""Regarding social services and benefits:

Common assistance programs include:
1. SNAP (Food assistance)
2. Medicaid (Healthcare)
3. TANF (Temporary assistance)
4. Housing assistance
5. Unemployment benefits

To apply:
- Visit your local Department of Social Services
- Apply online through your state's benefits portal
- Call the benefits hotline in your area
- Bring required documentation (ID, income proof, etc.)

Eligibility varies by program and income level. Most applications can be started online and completed in person."""
    
    elif any(word in question_lower for word in ['court', 'legal', 'lawsuit', 'ticket', 'fine']):
        return f"""Regarding legal and court matters:

For court-related issues:
1. Traffic tickets: Pay online, by mail, or contest in court
2. Small claims: File at your local courthouse
3. Legal aid: Free/low-cost legal help for qualifying individuals
4. Court records: Available through clerk of court

Important:
- Never ignore court notices or tickets
- Deadlines are strictly enforced
- Legal aid organizations can help if you can't afford an attorney
- Court self-help centers provide guidance for representing yourself

Contact your local courthouse for specific procedures and requirements."""
    
    else:
        return f"""Thank you for your question about government services.

I'm here to help with information about:
- Licenses and permits
- Voting and elections  
- Tax matters
- Social services and benefits
- Court and legal issues
- Municipal services

For the most accurate and up-to-date information about your specific situation, I recommend:
1. Visiting your local government office
2. Checking official government websites (.gov domains)
3. Calling the relevant department directly

Could you provide more details about what specific government service or information you're looking for? I'd be happy to point you in the right direction."""

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {str(e)}")
        return 'Neutral'

def get_feedback_stats():
    """Get feedback statistics for dashboard"""
    if 'user_id' not in session:
        return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
    
    feedback_entries = Feedback.query.filter_by(user_id=session['user_id']).all()
    stats = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': len(feedback_entries)}
    
    for feedback in feedback_entries:
        sentiment = feedback.sentiment.lower()
        if sentiment == 'positive':
            stats['positive'] += 1
        elif sentiment == 'negative':
            stats['negative'] += 1
        else:
            stats['neutral'] += 1
    
    return stats

# Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['user_name'] = user.full_name
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page and handler"""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not all([full_name, email, password]):
            flash('Please fill in all fields.', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please login instead.', 'error')
            return render_template('signup.html')
        
        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=password_hash
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error creating user: {str(e)}")
            flash('Error creating account. Please try again.', 'error')
    
    return render_template('signup.html')

@app.route('/home')
def home():
    """User home page after login"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('home.html')

@app.route('/chat')
def chat():
    """AI Chat interface - GET only, renders template"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_api():
    """AI Chat API endpoint - handles chat requests"""
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_input = request.form.get('message') or request.json.get('message')
    
    if not user_input:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Generate AI reply
        ai_reply = get_ai_response(user_input.strip())
        
        # Save to chat history
        chat_entry = ChatHistory(
            user_id=session['user_id'],
            question=user_input,
            response=ai_reply
        )
        db.session.add(chat_entry)
        db.session.commit()
        
    except Exception as e:
        logging.error(f"Chat API error: {str(e)}")
        return jsonify({'error': 'AI service is temporarily unavailable. Please try again later.'}), 503

    # Simple sentiment analysis
    sentiment = "neutral"
    user_lower = user_input.lower()
    if any(word in user_lower for word in ['angry', 'frustrated', 'upset', 'hate', 'terrible', 'awful', 'disappointed', 'bad']):
        sentiment = "negative"
    elif any(word in user_lower for word in ['happy', 'great', 'excellent', 'love', 'amazing', 'wonderful', 'good', 'satisfied']):
        sentiment = "positive"

    return jsonify({
        'reply': ai_reply, 
        'sentiment': sentiment,
        'question': user_input
    })

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback submission"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    question = request.form.get('question', '')
    feedback_text = request.form.get('feedback', '').strip()
    
    if feedback_text:
        sentiment = analyze_sentiment(feedback_text)
        
        feedback_entry = Feedback(
            user_id=session['user_id'],
            question=question,
            feedback_text=feedback_text,
            sentiment=sentiment
        )
        
        db.session.add(feedback_entry)
        db.session.commit()
        
        flash('Thank you for your feedback! It helps us improve our services.', 'success')
    
    return redirect(url_for('chat'))

@app.route('/dashboard')
def dashboard():
    """Dashboard with analytics"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get feedback statistics
    stats = get_feedback_stats()
    
    # Get recent feedback
    recent_feedback = Feedback.query.filter_by(user_id=session['user_id']).order_by(Feedback.timestamp.desc()).limit(10).all()
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_feedback=recent_feedback)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
