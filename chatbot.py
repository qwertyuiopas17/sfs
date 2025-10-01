from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import logging
import traceback
from datetime import datetime, timedelta
import json
import time
import threading
from typing import Dict, Any, Optional
import numpy as np
import atexit
# (In chatbot.py, after the import statements)
import re
from langdetect import detect, LangDetectException
from functools import wraps

# In chatbot.py, replace the clean_ai_response function

def clean_ai_response(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned_text = text.replace('\\n', '\n').replace("SAHARA:", "").strip()
    # Keep minimal filtering to avoid corrupting JSON-like content
    instructional_phrases = [
        "your task is to", "your response must be only", "return only json"
    ]
    lines = cleaned_text.splitlines()
    filtered = [ln for ln in lines if not any(p in ln.lower() for p in instructional_phrases)]
    return "\n".join(filtered).strip()
# Import enhanced database models
from enhanced_database_models import (
    db, User, Doctor, Appointment, HealthRecord, Pharmacy, MedicineOrder,
    ConversationTurn, UserSession, GrievanceReport, SystemMetrics, init_database, get_user_statistics
)

# Import enhanced AI components with Ollama integration
from nlu_processor import ProgressiveNLUProcessor
from ko import ProgressiveResponseGenerator
# Remove crisis detector import and usage
# from optimized_crisis_detector import OptimizedCrisisDetector  # removed

from api_ollama_integration import sehat_sahara_client, groq_scout

# Configure comprehensive logging with multiple handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handlers
file_handler = logging.FileHandler('chatbot.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

error_handler = logging.FileHandler('system_errors.log', mode='a', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Add console handler with proper encoding
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)

# Get module logger
logger = logging.getLogger(__name__)

# Initialize Flask application with enhanced configuration
app = Flask(__name__)
# Enhanced CORS configuration for production
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": [
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "https://saharasaathi.netlify.app",
            "https://sahara-sathi.onrender.com",
            "https://sahara-sathi.onrender.com/",
            "null",  # Allow null origin for local file testing
            "*"  # Allow all origins for development and static files
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", "X-CSRFToken"],
        "expose_headers": ["Content-Type", "X-CSRFToken"],
        "supports_credentials": True
    }
})


# Enhanced security configuration
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)

# Enhanced database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
models_path = os.path.join(basedir, 'models')
logs_path = os.path.join(basedir, 'logs')

# Ensure all directories exist
for path in [instance_path, models_path, logs_path]:
    os.makedirs(path, exist_ok=True)

# --- THIS IS THE CORRECTED CODE BLOCK ---

# First, determine the correct database URI
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    # This path is for Render (PostgreSQL)
    db_uri = database_url.replace('postgres://', 'postgresql://', 1)
else:
    # This path is for your local computer (SQLite)
    db_uri = f'sqlite:///{os.path.join(instance_path, "enhanced_chatbot.db")}'

# Now, update the app configuration
app.config.update({
    'SQLALCHEMY_DATABASE_URI': db_uri,
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'SQLALCHEMY_ENGINE_OPTIONS': {
        'pool_timeout': 30,
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'echo': False
    }
})

# Initialize database
db.init_app(app)

# Global system state tracking
system_state = {
    'startup_time': datetime.now(),
    'total_requests': 0,
    'successful_responses': 0,
    'error_count': 0,
    'appointments_booked': 0,
    'sos_triggered': 0,
    'llama_responses': 0,
    'fallback_responses': 0
}

# Thread lock for system state updates
state_lock = threading.Lock()

# WebRTC signaling storage
webrtc_signaling = {}  # {appointmentId: {'offers': [], 'answers': [], 'ice': []}}
webrtc_lock = threading.Lock()

# Initialize enhanced AI components with comprehensive error handling
def initialize_ai_components():
    """Initialize all AI components with Ollama integration and proper error handling"""
    global nlu_processor, response_generator, conversation_memory
    global system_status

    logger.info("🚀 Initializing Sehat Sahara Health Assistant...")

    # Model file paths
    nlu_model_path = os.path.join(models_path, 'progressive_nlu_model.pkl')
    memory_model_path = os.path.join(models_path, 'progressive_memory.pkl')

    system_status = {
        'nlu_processor': False,
        'response_generator': False,
        'conversation_memory': False,
        'database': False,
        'ollama_llama3': sehat_sahara_client.is_available  # reuse flag to show API availability
    }

    try:
        # Check Ollama Llama 3 availability (using Sehat Sahara client's flag)
        logger.info("🦙 Checking Sehat Sahara API availability...")
        system_status['ollama_llama3'] = sehat_sahara_client.is_available

        if sehat_sahara_client.is_available:
            logger.info("✅ Sehat Sahara API is available and ready for AI-enhanced responses")
        else:
            logger.info("⚠️ Sehat Sahara API not available - using rule-based responses with fallback")

        # Initialize NLU Processor
        logger.info("🧠 Initializing Progressive NLU Processor...")
        nlu_processor = ProgressiveNLUProcessor(model_path=nlu_model_path)
        system_status['nlu_processor'] = True
        logger.info("✅ NLU Processor initialized successfully")

        # Initialize Response Generator
        logger.info("💬 Initializing Progressive Response Generator...")
        response_generator = ProgressiveResponseGenerator()
        system_status['response_generator'] = True
        logger.info("✅ Response Generator initialized successfully")

        # Initialize Conversation Memory
        logger.info("Initializing Progressive Conversation Memory...")
        from conversation_memory import ProgressiveConversationMemory  # keep memory
        global conversation_memory
        conversation_memory = ProgressiveConversationMemory()
        system_status['conversation_memory'] = True
        logger.info("Conversation Memory initialized successfully")

        logger.info("✅ All AI components initialized for Sehat Sahara.")
        return True

    except Exception as e:
        logger.error(f"❌ Critical error initializing AI components: {e}")
        logger.error(traceback.format_exc())

        # Initialize minimal fallback components
        try:
            logger.info("🔄 Attempting to initialize fallback components...")
            nlu_processor = ProgressiveNLUProcessor()
            response_generator = ProgressiveResponseGenerator()
            from conversation_memory import ProgressiveConversationMemory
            conversation_memory = ProgressiveConversationMemory()
            logger.info("⚠️ Fallback components initialized (limited functionality)")
            return False
        except Exception as fallback_error:
            logger.error(f"❌ Failed to initialize even fallback components: {fallback_error}")
            nlu_processor = None
            response_generator = None
            conversation_memory = None
            return False

# Initialize system
ai_initialized = initialize_ai_components()

# Database initialization with app context
with app.app_context():
    try:
        init_database(app)
        system_status['database'] = True
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        system_status['database'] = False

# Utility functions for system management
def update_system_state(operation: str, success: bool = True, **kwargs):
    """Thread-safe system state updates"""
    with state_lock:
        system_state['total_requests'] += 1
        if success:
            system_state['successful_responses'] += 1
        else:
            system_state['error_count'] += 1

        # Update specific counters
        for key, value in kwargs.items():
            if key in system_state:
                system_state[key] += value

def save_all_models():
    """Save all AI models with comprehensive error handling"""
    try:
        if nlu_processor and system_status['nlu_processor']:
            nlu_processor.save_nlu_model(os.path.join(models_path, 'progressive_nlu_model.pkl'))
            logger.info("NLU model saved")

        if conversation_memory and system_status['conversation_memory']:
            conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
            logger.info("Conversation memory saved")

        # Crisis detector model saving removed as it's no longer used

        logger.info("All models saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def track_system_metrics():
    """Tracks and updates system-wide metrics for the Admin dashboard."""
    try:
        today = datetime.now().date()

        # Avoid creating duplicate metrics for the same day
        if SystemMetrics.query.filter_by(metrics_date=today).first():
            logger.info(f"Metrics for {today} already exist. Skipping.")
            return

        start_of_day = datetime.combine(today, datetime.min.time())

        # Calculate metrics for today
        active_users = User.query.filter(User.last_login >= start_of_day).count()
        new_users = User.query.filter(User.created_at >= start_of_day).count()
        total_convos = ConversationTurn.query.filter(ConversationTurn.timestamp >= start_of_day).count()
        appts_booked = Appointment.query.filter(Appointment.created_at >= start_of_day).count()
        orders_placed = MedicineOrder.query.filter(MedicineOrder.created_at >= start_of_day).count()

        # You would add grievances and prescriptions issued counts here as well

        metrics = SystemMetrics(
            metrics_date=today,
            total_active_users=active_users,
            new_users_registered=new_users,
            total_conversations=total_convos,
            appointments_booked=appts_booked,
            orders_placed=orders_placed
        )

        db.session.add(metrics)
        db.session.commit()
        logger.info(f"✅ Admin System Metrics updated for {today}")

    except Exception as e:
        logger.error(f"❌ Error tracking system metrics: {e}")
        db.session.rollback()

def get_current_user():
    """Security helper to get current authenticated user"""
    user_id = session.get('user_id')
    if user_id:
        try:
            # Fetch user with role information
            user = User.query.get(user_id)
            if user:
                # Dynamically add role if it exists on the model, otherwise default
                user.role = getattr(user, 'role', 'patient')
            return user
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None
    return None

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        # Assuming User model has a 'role' attribute, default to 'patient' if not present
        if getattr(user, "role", "patient") != "admin":
            return jsonify({"error": "Forbidden"}), 403
        return f(*args, **kwargs)
    return wrapper

def create_user_session(user: User, request_info: dict):
    """Create and track user session"""
    try:
        user_session = UserSession(
            user_id=user.id,
            ip_address=request_info.get('remote_addr', '')[:45],
            user_agent=request_info.get('user_agent', '')[:500],
            device_type=determine_device_type(request_info.get('user_agent', ''))
        )

        db.session.add(user_session)
        db.session.commit()

        # Store session ID for later reference
        session['session_record_id'] = user_session.id

    except Exception as e:
        logger.error(f"Error creating user session: {e}")

def determine_device_type(user_agent: str) -> str:
    """Determine device type from user agent"""
    user_agent = user_agent.lower()
    if any(mobile in user_agent for mobile in ['mobile', 'android', 'iphone']):
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    else:
        return 'desktop'

def end_user_session():
    """End current user session"""
    try:
        session_id = session.get('session_record_id')
        if session_id:
            user_session = UserSession.query.get(session_id)
            if user_session:
                user_session.end_session()
                db.session.commit()
    except Exception as e:
        logger.error(f"Error ending user session: {e}")

def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types in a dictionary."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Enhanced API Routes with comprehensive functionality
@app.route("/v1/register", methods=["POST"])
def register():
    try:
        update_system_state('register')
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided."}), 400

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        full_name = data.get("fullName", "").strip()
        phone_number = data.get("phoneNumber", "").strip()
        village = data.get("village", "").strip()
        district = data.get("district", "").strip()
        state = data.get("state", "Punjab").strip()
        pincode = data.get("pincode", "").strip()
        preferred_language = data.get("preferredLanguage", "hi").strip()

        # Validate required fields
        errors = []
        if not full_name: errors.append("Full name is required")
        if not email or "@" not in email: errors.append("Valid email is required")
        if not password or len(password) < 8: errors.append("Password must be at least 8 characters")
        if errors:
            return jsonify({"success": False, "message": "Fix the errors below.", "errors": errors}), 400

        # Check if the full name or email is already taken
        existing_user = User.query.filter((User.full_name == full_name) | (User.email == email)).first()
        if existing_user:
            return jsonify({"success": False, "message": "A user with this name or email already exists."}), 409

        # Auto-generate patient ID
        last_user = User.query.order_by(User.id.desc()).first()
        seq = last_user.id + 1 if last_user else 1
        patient_id = f"PAT{str(seq).zfill(6)}"

        new_user = User(
            patient_id=patient_id,
            email=email,
            full_name=full_name,
            phone_number=phone_number,
            village=village,
            district=district,
            state=state,
            pincode=pincode,
            preferred_language=preferred_language
        )
        new_user.set_password(password)
        # --- FIX: Explicitly set the role for new patient registrations to prevent DB errors ---
        new_user.role = 'patient'

        db.session.add(new_user)
        db.session.commit()

        logger.info(f"✅ New patient registered: {full_name} ({patient_id})")
        return jsonify({
            "success": True,
            "message": f"Welcome {full_name}! Your account has been created.",
            "patientId": patient_id,
            "fullName": full_name
        }), 201

    except Exception as e:
        # --- FIX: Added more detailed logging for registration errors ---
        logger.error(f"❌ Registration error for data: {data}. Exception: {e}")
        logger.error(traceback.format_exc())
        db.session.rollback()
        update_system_state('register', success=False)
        return jsonify({"success": False, "message": "An internal error occurred during registration. Please try again."}), 500

@app.route("/v1/login", methods=["POST", "OPTIONS"])
def login():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = app.make_response('')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin, X-CSRFToken')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response, 200

    try:
        update_system_state('login')
        data = request.get_json() or {}
        login_identifier = data.get("patientId", "").strip()
        password = data.get("password", "")
        role = data.get("role", "patient").strip().lower()

        if not login_identifier or not password:
            return jsonify({"success": False, "message": "Username and password are required."}), 400

        # Role-based authentication
        if role == 'doctor':
            # Query Doctor table
            doctor = None
            logger.info(f"Doctor login attempt: identifier='{login_identifier}', role='{role}'")
            if '@' in login_identifier:
                doctor = Doctor.query.filter_by(email=login_identifier.lower()).first()
                logger.info(f"Queried by email: {login_identifier.lower()}, found: {doctor is not None}")
            elif login_identifier.upper().startswith('DOC'):
                doctor = Doctor.query.filter_by(doctor_id=login_identifier.upper()).first()
                logger.info(f"Queried by doctor_id: {login_identifier.upper()}, found: {doctor is not None}")
            else:
                logger.info(f"Identifier doesn't start with DOC or contain @: {login_identifier}")

            # Fallback to check by name
            if not doctor:
                doctor = Doctor.query.filter_by(full_name=login_identifier).first()
                logger.info(f"Fallback query by name: {login_identifier}, found: {doctor is not None}")

            if doctor:
                logger.info(f"Doctor found: id={doctor.id}, active={doctor.is_active}, has_password={doctor.password_hash is not None}")
                if doctor.is_active:
                    password_check = doctor.check_password(password)
                    logger.info(f"Password check result: {password_check}")
                    if password_check:
                        # For doctors, create a session with doctor info
                        session.permanent = True
                        session['user_id'] = doctor.id
                        session['doctor_id'] = doctor.doctor_id
                        session['role'] = 'doctor'
                        session['login_time'] = datetime.now().isoformat()

                        # Create user session (reuse for doctors)
                        create_user_session(doctor, {
                            'remote_addr': request.environ.get('REMOTE_ADDR'),
                            'user_agent': request.environ.get('HTTP_USER_AGENT')
                        })

                        logger.info(f"✅ Doctor login successful: {doctor.full_name} ({doctor.doctor_id})")
                        return jsonify({
                            "success": True,
                            "message": f"Welcome back, Dr. {doctor.full_name}!",
                            "user": {
                                "patientId": doctor.doctor_id,  # Use doctor_id as patientId for consistency
                                "username": doctor.full_name,
                                "email": doctor.email,
                                "role": "doctor"
                            }
                        })
                    else:
                        logger.warning(f"Password check failed for doctor {doctor.doctor_id}")
                else:
                    logger.warning(f"Doctor account inactive: {doctor.doctor_id}")
                    return jsonify({"success": False, "message": "This account is inactive."}), 403
            else:
                logger.warning(f"Doctor not found for identifier: {login_identifier}")
        else:
            # Default to patient login
            # --- FIX: Enhanced query to find user by email, patient_id, or full_name ---
            user = None
            if '@' in login_identifier:
                user = User.query.filter_by(email=login_identifier.lower()).first()
            elif login_identifier.upper().startswith('PAT'):
                user = User.query.filter_by(patient_id=login_identifier.upper()).first()

            # Fallback to check by name if not found by email or ID
            if not user:
                user = User.query.filter_by(full_name=login_identifier).first()

            if user and user.is_active and user.check_password(password):
                user.update_last_login()
                db.session.commit()
                session.permanent = True
                session['user_id'] = user.id
                session['patient_id'] = user.patient_id
                session['role'] = 'patient'
                session['login_time'] = datetime.now().isoformat()

                create_user_session(user, {
                    'remote_addr': request.environ.get('REMOTE_ADDR'),
                    'user_agent': request.environ.get('HTTP_USER_AGENT')
                })

                stats = get_user_statistics(user.id) or {}

                logger.info(f"✅ User login successful: {user.full_name} ({user.patient_id}) with role '{user.role}'")
                # --- FIX: Return user role to frontend for proper redirection ---
                return jsonify({
                    "success": True,
                    "message": f"Welcome back, {user.full_name}!",
                    "user": {
                        "patientId": user.patient_id,
                        "username": user.full_name,
                        "email": user.email,
                        "role": user.role # This is crucial for the frontend
                    },
                    "statistics": stats
                })
            if user and not user.is_active:
                  return jsonify({"success": False, "message": "This account is inactive."}), 403

        logger.warning(f"⚠️ Failed login attempt for identifier: '{login_identifier}' as {role}")
        update_system_state('login', success=False)
        return jsonify({"success": False, "message": "Invalid credentials or account inactive."}), 401

    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('login', success=False)
        return jsonify({"success": False, "message": "Login failed due to a server error."}), 500


@app.route("/v1/logout", methods=["POST"])
def logout():
    """Enhanced user logout with session cleanup"""
    try:
        # End user session tracking
        end_user_session()

        # Clear session
        session.clear()

        logger.info("✅ User logged out successfully")

        return jsonify({
            "success": True,
            "message": "You have been logged out successfully."
        })

    except Exception as e:
        logger.error(f"❌ Logout error: {e}")
        return jsonify({
            "success": True,  # Always succeed logout for security
            "message": "Logged out successfully."
        })

@app.route("/v1/predict", methods=["POST"])
def predict():
    if not all([nlu_processor, response_generator, conversation_memory]):
        return jsonify({"error": True, "message": "AI components unavailable"}), 503

    data = {}
    try:
        start_time = time.time()
        update_system_state('predict')

        data = request.get_json() or {}
        logger.info(f"Request data: {data}")
        user_message = (data.get("message") or "").strip()
        image_b64 = data.get("imageData")
        user_id_str = (data.get("userId") or "").strip()
        context = data.get("context", {}) or {}

        # Add logging to track incoming requests
        logger.info(f"🔍 /v1/predict called - userId: '{user_id_str}', message: '{user_message[:100]}...'")

        if not user_id_str:
            return jsonify({"error": "User ID is required."}), 400
        if not user_message and not image_b64:
            return jsonify({"error": "Either message or imageData is required."}), 400

        # Load current user by patient_id
        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        logger.info(f"User lookup: user_id_str='{user_id_str}', current_user={current_user is not None}")
        if not current_user:
            return jsonify({"error": "User not found.", "login_required": True}), 401

        # Detect language from message
        detected_language_code = "en"
        if user_message:
            try:
                lang = detect(user_message)
                detected_language_code = 'hi' if lang == 'hi' else ('pa' if lang == 'pa' else 'en')
            except LangDetectException:
                detected_language_code = current_user.preferred_language or "en"

        # --- FIX #1 START ---
        # Build short NLU history using the correct method
        history_turns = conversation_memory.get_conversation_context(current_user.patient_id, turns=4)
        nlu_history = []
        for turn in history_turns:
            nlu_history.append({'role': 'user', 'content': turn.get('user_message', '')})
            # The bot_response from memory is a JSON string, so we parse it to get the text part
            try:
                bot_response_json = json.loads(turn.get('bot_response', '{}'))
                bot_content = bot_response_json.get('response', '')
            except (json.JSONDecodeError, AttributeError):
                bot_content = turn.get('bot_response', '') # Fallback for non-JSON responses
            nlu_history.append({'role': 'assistant', 'content': bot_content})
        # --- FIX #1 END ---

        # Optional Scout for emojis/images
        effective_message = user_message
        scout_text = None
        contains_emoji = bool(re.search(r"[\U0001F300-\U0001FAFF\U00002600-\U000026FF]", user_message))
        multimodal_triggered = bool(image_b64) or contains_emoji
        if multimodal_triggered and groq_scout and groq_scout.is_available:
            if image_b64:
                scout_text = groq_scout.interpret_image(user_message=user_message, image_b64=image_b64, language=detected_language_code, context_history=nlu_history)
            else:
                scout_text = groq_scout.interpret_emojis(user_message=user_message, language=detected_language_code, context_history=nlu_history)
            effective_message = f"Interpreted content: {scout_text}\n\nOriginal: {user_message}" if scout_text else user_message

        # NLU: Sehat Sahara intents
        nlu_understanding = nlu_processor.understand_user_intent(effective_message, conversation_history=nlu_history)

        # Emergency handling protocol
        is_emergency = (nlu_understanding.get('primary_intent') == 'emergency_assistance') or (nlu_understanding.get('urgency_level') == 'emergency')
        if is_emergency:
            action_payload = {
                "response": "Emergency detected. Connecting to emergency services. For ambulance, call 108.",
                "action": "TRIGGER_SOS",
                "parameters": {"emergency_number": "108", "type": "medical_emergency"}
            }
            update_system_state('predict', sos_triggered=1)
        else:
            action_payload_str = None
            action_payload = None
            if sehat_sahara_client and sehat_sahara_client.is_available:
                # --- FIX #2 START ---
                # Use the correct method name: generate_sehatsahara_response
                action_payload_str = sehat_sahara_client.generate_sehatsahara_response(
                    user_message=effective_message,
                    user_intent=nlu_understanding.get('primary_intent'),
                    conversation_stage=nlu_understanding.get('conversation_stage'),
                    severity_score=0.5, # Placeholder, you can develop this
                    context_history=nlu_history,
                    language=detected_language_code
                )
                # --- FIX #2 END ---
                if action_payload_str:
                    try:
                        action_payload = json.loads(action_payload_str)
                    except json.JSONDecodeError:
                        logger.warning("API returned invalid JSON. Falling back to local generator.")
                        action_payload = None

            # Fallback to local response generator if API unavailable or fails
            if not action_payload:
                update_system_state('predict', fallback_responses=1)
                action_payload = response_generator.generate_response(
                    user_message=effective_message,
                    nlu_result={**nlu_understanding, "language_detected": detected_language_code},
                    user_context={"user_id": current_user.patient_id, "session_id": session.get('session_record_id')},
                    conversation_history=nlu_history
                )

        # Clean only the response text
        if action_payload.get("response"):
            action_payload["response"] = clean_ai_response(action_payload["response"])

        # Save conversation turn with action fields
        turn = ConversationTurn(
            user_id=current_user.id,
            user_message=effective_message,
            bot_response=json.dumps(action_payload), # Store the whole JSON payload
            detected_intent=nlu_understanding.get('primary_intent'),
            intent_confidence=nlu_understanding.get('confidence', 0.5),
            language_detected=detected_language_code,
            urgency_level=nlu_understanding.get('urgency_level', 'low'),
            response_time_ms=int((time.time() - start_time) * 1000),
            action_triggered=action_payload.get("action"),
        )
        turn.set_action_parameters(action_payload.get("parameters", {}))
        turn.set_context_entities(nlu_understanding.get('context_entities', {}))
        db.session.add(turn)

        # Update session counters
        try:
            session_record_id = session.get('session_record_id')
            if session_record_id:
                user_session = UserSession.query.get(session_record_id)
                if user_session:
                    user_session.conversations_in_session += 1
                    user_session.actions_triggered_in_session += 1
        except Exception:
            pass

        db.session.commit()

        # Final envelope including analysis metadata
        enriched = {
            **action_payload,
            "analysis": {
                "intent": nlu_understanding.get('primary_intent'),
                "confidence": nlu_understanding.get('confidence', 0.5),
                "language": detected_language_code,
                "urgency": nlu_understanding.get('urgency_level', 'low'),
                "in_scope": nlu_understanding.get('in_scope', True),
            },
            "system_info": {
                "response_time_ms": int((time.time() - start_time) * 1000),
                "api_available": bool(sehat_sahara_client and sehat_sahara_client.is_available)
            }
        }
        return jsonify(enriched)

    except Exception as e:
        logger.error(f"Error in predict for user {data.get('userId', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        update_system_state('predict', success=False)

        error_response = {
            "error": True,
            "message": "Unable to process message at the moment.",
            "fallback_resources": {
                "emergency_services": "108",
                "health_helpline": "104"
            }
        }
        return jsonify(error_response), 500

# (deleted)

@app.route("/v1/book-doctor", methods=["POST"])
def book_doctor():
    try:
        update_system_state('book_doctor')
        data = request.get_json() or {}
        user_id_str = (data.get("userId") or "").strip()
        doctor_id_str = str(data.get("doctorId") or "").strip()
        appointment_dt = (data.get("appointmentDatetime") or "").strip()
        appointment_type = (data.get("appointmentType") or "consultation").strip()
        chief_complaint = (data.get("chiefComplaint") or "").strip()
        symptoms = data.get("symptoms") or []

        if not user_id_str or not doctor_id_str or not appointment_dt:
            return jsonify({"error": "userId, doctorId and appointmentDatetime are required"}), 400

        user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not user:
            return jsonify({"error": "User not found"}), 401

        doctor = Doctor.query.filter(
            (Doctor.doctor_id == doctor_id_str) | (Doctor.id == doctor_id_str)
        ).first()
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        # Parse datetime in ISO format
        try:
            when = datetime.fromisoformat(appointment_dt)
        except Exception:
            return jsonify({"error": "Invalid appointmentDatetime. Use ISO 8601 format."}), 400

        appt = Appointment(
            user_id=user.id,
            doctor_id=doctor.id,
            appointment_datetime=when,
            appointment_type=appointment_type,
            chief_complaint=chief_complaint
        )
        appt.set_symptoms(symptoms)
        db.session.add(appt)

        # Update session counters
        try:
            session_record_id = session.get('session_record_id')
            if session_record_id:
                s = UserSession.query.get(session_record_id)
                if s:
                    s.appointments_booked_in_session += 1
        except Exception:
            pass

        db.session.commit()
        update_system_state('book_doctor', appointments_booked=1)

        return jsonify({
            "success": True,
            "message": "Appointment created.",
            "appointment": {
                "appointmentId": appt.appointment_id,
                "doctor": {"id": doctor.doctor_id, "name": doctor.full_name, "specialization": doctor.specialization},
                "datetime": appt.appointment_datetime.isoformat(),
                "type": appt.appointment_type,
                "status": appt.status
            }
        })
    except Exception as e:
        logger.error(f"Book doctor error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('book_doctor', success=False)
        return jsonify({"error": "Failed to book appointment"}), 500

@app.route("/v1/history", methods=["POST"])
def get_history():
    """Get comprehensive conversation history with analytics"""
    try:
        update_system_state('get_history')
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_id_str = data.get("userId", "")
        limit = min(data.get("limit", 50), 100)
        include_analysis = data.get("includeAnalysis", False)

        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400

        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401

        # Get conversation turns with ordering
        turns = ConversationTurn.query.filter_by(user_id=current_user.id)\
                .order_by(ConversationTurn.timestamp.asc())\
                .limit(limit).all()

        # Format conversation history
        chat_log = []
        for turn in turns:
            # User message
            user_entry = {
                "role": "user",
                "content": turn.user_message,
                "timestamp": turn.timestamp.isoformat(),
                "turn_id": turn.id
            }

            # Assistant response with optional analysis
            assistant_entry = {
                "role": "assistant",
                "content": turn.bot_response,
                "timestamp": turn.timestamp.isoformat(),
                "turn_id": turn.id
            }

            if include_analysis:
                assistant_entry["analysis"] = {
                    "intent": turn.detected_intent,
                    "confidence": turn.intent_confidence,
                    "language": turn.language_detected,
                    "urgency": turn.urgency_level,
                    "action": turn.action_triggered,
                    "action_parameters": turn.get_action_parameters(),
                    "context_entities": turn.get_context_entities()
                }

            chat_log.extend([user_entry, assistant_entry])

        # Get user progress summary
        user_summary = conversation_memory.get_user_summary(user_id_str) if conversation_memory else {}

        response_data = {
            "success": True,
            "history": chat_log,
            "summary": {
                "total_conversations": current_user.total_conversations,
                "current_stage": current_user.current_conversation_stage,
                "risk_level": current_user.current_risk_level,
                "improvement_trend": current_user.improvement_trend,
                "member_since": current_user.created_at.isoformat(),
                "last_interaction": current_user.last_login.isoformat() if current_user.last_login else None
            }
        }

        # Add detailed progress if available
        if user_summary.get('exists'):
            response_data["progress_analytics"] = user_summary.get('progress_metrics', {})
            response_data["method_analytics"] = user_summary.get('method_effectiveness', {})
            response_data["risk_assessment"] = user_summary.get('risk_assessment', {})

        logger.info(f"✅ History retrieved for {user_id_str}: {len(turns)} turns")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ History retrieval error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('get_history', success=False)

        return jsonify({
            "error": "Failed to retrieve conversation history",
            "message": "Please try again later"
        }), 500

@app.route("/v1/user-stats", methods=["POST"])
def get_user_stats():
    try:
        update_system_state('get_user_stats')
        data = request.get_json() or {}
        user_id_str = (data.get("userId") or "").strip()
        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400
        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401

        stats = get_user_statistics(current_user.id) or {}
        return jsonify({"success": True, **stats})
    except Exception as e:
        logger.error(f"User stats error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('get_user_stats', success=False)
        return jsonify({"error": "Failed to retrieve user statistics"}), 500

@app.route("/v1/health", methods=["GET"])
def health_check():
    """Comprehensive system health check with Ollama status"""
    try:
        # System health overview
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system_state['startup_time']).total_seconds() / 3600, 2),
            "components": system_status,
            "api_ollama_integration": {
                "available": sehat_sahara_client.is_available,
                "status": "connected" if sehat_sahara_client.is_available else "fallback_mode"
            },
            "system_metrics": {
                "total_requests": system_state['total_requests'],
                "successful_responses": system_state['successful_responses'],
                "error_count": system_state['error_count'],
                "success_rate": system_state['successful_responses'] / max(system_state['total_requests'], 1),
                "appointments_booked": system_state.get('appointments_booked', 0),
                "sos_triggered": system_state.get('sos_triggered', 0),
                "llama_responses": system_state.get('llama_responses', 0),
                "fallback_responses": system_state.get('fallback_responses', 0)
            }
        }

        # Database health check
        try:
            with app.app_context():
                total_users = User.query.count()
                total_conversations = ConversationTurn.query.count()
                health_status["database"] = {
                    "status": "connected",
                    "total_users": total_users,
                    "total_conversations": total_conversations
                }
        except Exception as e:
            health_status["database"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # Overall health assessment
        critical_components = ['nlu_processor', 'response_generator', 'conversation_memory', 'database']
        if not all(system_status.get(comp, False) for comp in critical_components):
            health_status["status"] = "degraded"

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route("/v1/admin/metrics", methods=["GET"])
@admin_required
def get_system_metrics():
    """Provides system-wide metrics for the admin dashboard."""
    try:
        # Get last 30 days of metrics
        thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
        recent_metrics = SystemMetrics.query.filter(SystemMetrics.metrics_date >= thirty_days_ago).order_by(SystemMetrics.metrics_date.desc()).all()

        metrics_data = [
            {
                "date": metric.metrics_date.isoformat(),
                "activeUsers": metric.total_active_users,
                "newUsers": metric.new_users_registered,
                "conversations": metric.total_conversations,
                "appointments": metric.appointments_booked,
                "orders": metric.orders_placed
            } for metric in recent_metrics
        ]

        # Get current high-level stats
        current_stats = {
            "totalUsers": User.query.count(),
            "totalDoctors": Doctor.query.count(),
            "totalPharmacies": Pharmacy.query.count(),
            "pendingGrievances": GrievanceReport.query.filter_by(status='Pending').count()
        }

        return jsonify({
            "success": True,
            "historicalMetrics": metrics_data,
            "currentStats": current_stats
        })

    except Exception as e:
        logger.error(f"❌ Metrics retrieval error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve system metrics"}), 500

@app.route("/v1/save-models", methods=["POST"])
def save_models_endpoint():
    """Manually trigger comprehensive model saving"""
    try:
        # Authentication check (basic - enhance in production)
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('ADMIN_API_KEY', 'admin_key_123'):
            return jsonify({"error": "Unauthorized"}), 401

        success = save_all_models()

        if success:
            return jsonify({
                "success": True,
                "message": "All AI models saved successfully",
                "timestamp": datetime.now().isoformat(),
                "models_saved": {
                    "nlu_processor": system_status['nlu_processor'],
                    "conversation_memory": system_status['conversation_memory'],
                    # Crisis detector model saving status removed
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Some models failed to save - check logs for details",
                "timestamp": datetime.now().isoformat()
            }), 500

    except Exception as e:
        logger.error(f"❌ Model saving endpoint error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving models: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/v1/system/status", methods=["GET"])
def system_status_endpoint():
    """Get detailed system status information"""
    return jsonify({
        "system_info": {
            "application_name": "Sehat Sahara Health Assistant",
            "version": "2.1.0",
            "startup_time": system_state['startup_time'].isoformat(),
            "current_time": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system_state['startup_time']).total_seconds() / 3600, 2)
        },
        "components": system_status,
        "api_ollama_integration": {
            "available": sehat_sahara_client.is_available,
            "model": sehat_sahara_client.client.model if sehat_sahara_client.is_available else "N/A",
            "base_url": sehat_sahara_client.client.base_url if sehat_sahara_client.is_available else "N/A"
        },
        "features": {
            "progressive_conversation_stages": True,
            "method_effectiveness_tracking": True,
            # Optimized crisis detection removed
            "professional_counselor_integration": False, # Removed as it's not relevant for Sehat Sahara
            "comprehensive_user_analytics": True,
            "persistent_conversation_memory": True,
            "enhanced_emotional_analysis": False, # Removed as it's not the focus
            "real_time_system_monitoring": True,
            "ollama_ai_enhancement": sehat_sahara_client.is_available,
            "task_oriented_actions": True, # New feature for Sehat Sahara
            "emergency_handling": True # New feature for Sehat Sahara
        },
        "performance": {
            "total_requests": system_state['total_requests'],
            "successful_responses": system_state['successful_responses'],
            "error_count": system_state['error_count'],
            "success_rate": system_state['successful_responses'] / max(system_state['total_requests'], 1),
            "appointments_booked": system_state.get('appointments_booked', 0),
            "sos_triggered": system_state.get('sos_triggered', 0),
            "llama_responses": system_state.get('llama_responses', 0),
            "fallback_responses": system_state.get('fallback_responses', 0)
        }
    })

# Ollama-specific endpoints
@app.route("/v1/ollama/status", methods=["GET"])
def ollama_status():
    """Get Ollama integration status"""
    return jsonify({
        "ollama_available": sehat_sahara_client.is_available,
        "client_info": {
            "base_url": sehat_sahara_client.client.base_url,
            "model": sehat_sahara_client.client.model
        },
        "integration_working": system_status.get('ollama_llama3', False),
        "responses_generated": system_state.get('llama_responses', 0),
        "fallback_responses": system_state.get('fallback_responses', 0)
    })

@app.route("/v1/ollama/test", methods=["POST"])
def test_ollama():
    try:
        data = request.get_json() or {}
        test_message = data.get("message", "Book a doctor for tomorrow morning in my village").strip()
        if sehat_sahara_client and sehat_sahara_client.is_available:
            result = sehat_sahara_client.generate_action_json(test_message)
            if result:
                return jsonify({"success": True, "result": result})
            return jsonify({"success": False, "message": "No result"}), 500
        return jsonify({"success": False, "message": "API not available"}), 503
    except Exception as e:
        logger.error(f"Ollama test error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/v1/scan-medicine", methods=["POST"])
def scan_medicine():
    """Scan medicine using Groq Scout AI"""
    try:
        data = request.get_json() or {}
        image_b64 = data.get("imageData")
        user_message = data.get("message", "Scan this medicine").strip()

        if not image_b64:
            return jsonify({"error": "imageData is required"}), 400

        if not groq_scout or not groq_scout.is_available:
            return jsonify({"error": "AI scanning service unavailable"}), 503

        # Use Groq Scout to interpret the medicine image
        medicine_info = groq_scout.interpret_image(
            user_message=user_message,
            image_b64=image_b64,
            language="en",  # Default to English for medicine scanning
            context_history=[]  # No context needed for standalone scan
        )

        if medicine_info:
            return jsonify({
                "success": True,
                "medicine_info": medicine_info,
                "scanned_at": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Could not interpret the medicine image"}), 500

    except Exception as e:
        logger.error(f"Scan medicine error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to scan medicine"}), 500

@app.route('/v1/user/<user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile and health app statistics"""
    try:
        user_summary = conversation_memory.get_user_summary(user_id)
        if not user_summary:
            return jsonify({
                "error": "User not found",
                "status": "error"
            }), 404

        return jsonify({
            "profile": user_summary,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to retrieve user profile",
            "status": "error"
        }), 500

@app.route('/v1/user/<user_id>/preferences', methods=['POST'])
def update_user_preferences(user_id):
    """Update user preferences"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No preferences data provided",
                "status": "error"
            }), 400

        # Update preferences in conversation memory
        conversation_memory.update_user_preferences(user_id, data)

        return jsonify({
            "message": "Preferences updated successfully",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to update preferences",
            "status": "error"
        }), 500

@app.route('/v1/stats', methods=['GET'])
def get_system_stats():
    """Get Sehat Sahara system statistics"""
    try:
        # Get conversation memory stats
        memory_stats = conversation_memory.get_system_stats()

        # Get database stats
        db_stats = {
            "total_users": User.query.count(),
            "active_users": User.query.filter_by(is_active=True).count(),
            "total_conversations": ConversationTurn.query.count(),
            "conversations_today": ConversationTurn.query.filter(
                ConversationTurn.timestamp >= datetime.now().date()
            ).count(),
            "emergency_calls_today": ConversationTurn.query.filter(
                ConversationTurn.urgency_level == 'emergency',
                ConversationTurn.timestamp >= datetime.now().date()
            ).count()
        }

        return jsonify({
            "system_stats": {
                "memory": memory_stats,
                "database": db_stats,
                "api_status": "available" if sehat_sahara_client.is_available else "unavailable",
                "service_name": "Sehat Sahara Health Assistant",
                "version": "2.1.0"
            },
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to retrieve system statistics",
            "status": "error"
        }), 500

# New Dashboard and Patient Management Endpoints
@app.route("/v1/dashboard", methods=["GET"])
def get_dashboard():
    """Get patient dashboard data"""
    try:
        update_system_state('get_dashboard')

        # Get user from session
        current_user = get_current_user()
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401

        # Get next appointment
        next_appointment = Appointment.query.filter(
            Appointment.user_id == current_user.id,
            Appointment.status == 'scheduled',
            Appointment.appointment_datetime > datetime.now()
        ).order_by(Appointment.appointment_datetime.asc()).first()

        next_appointment_data = None
        if next_appointment:
            doctor = Doctor.query.get(next_appointment.doctor_id)
            next_appointment_data = {
                "doctorName": doctor.full_name if doctor else "Unknown Doctor",
                "dateTime": next_appointment.appointment_datetime.isoformat()
            }

        # Get prescription count (count of completed appointments with prescriptions)
        prescription_count = Appointment.query.filter(
            Appointment.user_id == current_user.id,
            Appointment.status == 'completed',
            Appointment.prescription.isnot(None)
        ).count()

        # Get recent health records
        health_records = HealthRecord.query.filter_by(user_id=current_user.id)\
            .order_by(HealthRecord.created_at.desc()).limit(5).all()

        health_records_data = []
        for record in health_records:
            health_records_data.append({
                "date": record.created_at.isoformat(),
                "title": record.title,
                "description": record.description or ""
            })

        return jsonify({
            "success": True,
            "fullName": current_user.full_name,
            "nextAppointment": next_appointment_data,
            "prescriptionCount": prescription_count,
            "healthRecords": health_records_data
        })

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        update_system_state('get_dashboard', success=False)
        return jsonify({"success": False, "message": "Failed to load dashboard"}), 500

@app.route("/v1/doctors", methods=["GET"])
def get_doctors():
    """Get list of doctors with optional specialty filter"""
    try:
        update_system_state('get_doctors')

        specialty = request.args.get('specialty', '').strip()

        query = Doctor.query.filter_by(is_active=True)

        if specialty:
            query = query.filter(Doctor.specialization.ilike(f'%{specialty}%'))

        doctors = query.all()

        doctors_data = []
        for doctor in doctors:
            languages = doctor.get_languages_spoken()
            languages_str = ", ".join(languages) if languages else "English"
            doctors_data.append({
                "id": doctor.doctor_id,
                "name": doctor.full_name,
                "specialization": doctor.specialization,
                "qualification": doctor.qualification,
                "experience": doctor.experience_years,
                "clinicName": doctor.clinic_name,
                "rating": doctor.average_rating,
                "languages": languages_str,
                "img": doctor.profile_image_url or "https://placehold.co/100x100/0d9488/FFFFFF?text=Doc"
            })

        return jsonify({
            "success": True,
            "doctors": doctors_data,
            "count": len(doctors_data)
        })

    except Exception as e:
        logger.error(f"Get doctors error: {e}")
        update_system_state('get_doctors', success=False)
        return jsonify({"success": False, "message": "Failed to retrieve doctors"}), 500

@app.route("/v1/appointments", methods=["GET"])
def get_appointments():
    """Get user's appointments"""
    try:
        update_system_state('get_appointments')

        current_user = get_current_user()
        if not current_user:
            # Allow userId in query param for frontend compatibility
            user_id_str = request.args.get('userId')
            if user_id_str:
                current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
            if not current_user:
                return jsonify({"success": False, "message": "Authentication required"}), 401

        appointments = Appointment.query.filter_by(user_id=current_user.id)\
            .order_by(Appointment.appointment_datetime.desc()).all()

        appointments_data = []
        for appt in appointments:
            doctor = Doctor.query.get(appt.doctor_id)
            appointments_data.append({
                "id": appt.appointment_id,
                "doctorName": doctor.full_name if doctor else "Unknown Doctor",
                "specialization": doctor.specialization if doctor else "",
                "dateTime": appt.appointment_datetime.isoformat(),
                "type": appt.appointment_type,
                "status": appt.status,
                "chiefComplaint": appt.chief_complaint
            })

        return jsonify({
            "success": True,
            "appointments": appointments_data
        })

    except Exception as e:
        logger.error(f"Get appointments error: {e}")
        update_system_state('get_appointments', success=False)
        return jsonify({"success": False, "message": "Failed to retrieve appointments"}), 500

@app.route("/v1/appointments/cancel", methods=["POST"])
def cancel_appointment():
    """Cancel an appointment"""
    try:
        update_system_state('cancel_appointment')

        current_user = get_current_user()
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401

        data = request.get_json() or {}
        appointment_id = data.get("appointmentId", "").strip()

        if not appointment_id:
            return jsonify({"success": False, "message": "Appointment ID is required"}), 400

        appointment = Appointment.query.filter_by(
            appointment_id=appointment_id,
            user_id=current_user.id
        ).first()

        if not appointment:
            return jsonify({"success": False, "message": "Appointment not found"}), 404

        if appointment.cancel_appointment("Cancelled by user"):
            db.session.commit()
            return jsonify({
                "success": True,
                "message": "Appointment cancelled successfully"
            })
        else:
            return jsonify({"success": False, "message": "Appointment cannot be cancelled"}), 400

    except Exception as e:
        logger.error(f"Cancel appointment error: {e}")
        update_system_state('cancel_appointment', success=False)
        return jsonify({"success": False, "message": "Failed to cancel appointment"}), 500

@app.route("/v1/pharmacies", methods=["GET"])
def get_pharmacies():
    """Get list of pharmacies"""
    try:
        update_system_state('get_pharmacies')

        pharmacies = Pharmacy.query.filter_by(is_active=True).all()

        pharmacies_data = []
        for pharmacy in pharmacies:
            pharmacies_data.append({
                "id": pharmacy.pharmacy_id,
                "name": pharmacy.name,
                "address": pharmacy.address,
                "phone": pharmacy.phone_number,
                "services": {
                    "homeDelivery": pharmacy.home_delivery,
                    "onlinePayment": pharmacy.online_payment,
                    "emergencyService": pharmacy.emergency_service
                },
                "rating": pharmacy.average_rating
            })

        return jsonify({
            "success": True,
            "pharmacies": pharmacies_data,
            "count": len(pharmacies_data)
        })

    except Exception as e:
        logger.error(f"Get pharmacies error: {e}")
        update_system_state('get_pharmacies', success=False)
        return jsonify({"success": False, "message": "Failed to retrieve pharmacies"}), 500

@app.route("/v1/medicines", methods=["GET"])
def search_medicines():
    """Search for medicines"""
    try:
        update_system_state('search_medicines')

        search_term = request.args.get('search', '').strip()

        if not search_term:
            return jsonify({"success": False, "message": "Search term is required"}), 400

        # For now, return mock medicine data since we don't have a medicines table
        # In production, this would query a Medicine table
        mock_medicines = [
            {"name": "Paracetamol", "type": "Pain Relief", "dosage": "500mg", "price": 25.0},
            {"name": "Ibuprofen", "type": "Anti-inflammatory", "dosage": "400mg", "price": 35.0},
            {"name": "Amoxicillin", "type": "Antibiotic", "dosage": "500mg", "price": 85.0},
            {"name": "Cetirizine", "type": "Antihistamine", "dosage": "10mg", "price": 15.0},
            {"name": "Omeprazole", "type": "Antacid", "dosage": "20mg", "price": 45.0}
        ]

        # Filter medicines based on search term
        filtered_medicines = [
            med for med in mock_medicines
            if search_term.lower() in med["name"].lower()
        ]

        return jsonify({
            "success": True,
            "medicines": filtered_medicines,
            "count": len(filtered_medicines)
        })

    except Exception as e:
        logger.error(f"Search medicines error: {e}")
        update_system_state('search_medicines', success=False)
        return jsonify({"success": False, "message": "Failed to search medicines"}), 500

@app.route("/v1/orders", methods=["POST"])
def place_order():
    """Place a medicine order"""
    try:
        update_system_state('place_order')

        current_user = get_current_user()
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401

        data = request.get_json() or {}
        pharmacy_id = data.get("pharmacyId", "").strip()
        items = data.get("items", [])
        delivery_address = data.get("deliveryAddress", "").strip()

        if not pharmacy_id or not items:
            return jsonify({"success": False, "message": "Pharmacy ID and items are required"}), 400

        pharmacy = Pharmacy.query.filter_by(pharmacy_id=pharmacy_id, is_active=True).first()
        if not pharmacy:
            return jsonify({"success": False, "message": "Pharmacy not found"}), 404

        # Calculate total amount (mock calculation)
        total_amount = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
        delivery_fee = 50.0 if pharmacy.home_delivery else 0.0

        order = MedicineOrder(
            user_id=current_user.id,
            pharmacy_id=pharmacy.id,
            items=items,
            total_amount=total_amount,
            delivery_fee=delivery_fee,
            delivery_address=delivery_address or current_user.get_full_address(),
            contact_phone=current_user.phone_number
        )

        db.session.add(order)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Order placed successfully",
            "orderId": order.order_id,
            "estimatedDelivery": "30-45 mins"
        })

    except Exception as e:
        logger.error(f"Place order error: {e}")
        update_system_state('place_order', success=False)
        return jsonify({"success": False, "message": "Failed to place order"}), 500

@app.route("/v1/admin/users", methods=["GET"])
@admin_required
def get_all_users():
    """Admin endpoint to get all user types except patients."""
    try:
        doctors = Doctor.query.all()
        pharmacies = Pharmacy.query.all()

        all_users = []
        for d in doctors:
            all_users.append({"id": d.id, "name": d.full_name, "role": "Doctor", "contact": d.email, "status": "Verified" if d.is_verified else "Unverified"})
        for p in pharmacies:
            all_users.append({"id": p.id, "name": p.name, "role": "Pharmacy", "contact": p.phone_number, "status": "Verified" if p.is_verified else "Unverified"})

        return jsonify({"success": True, "users": all_users})
    except Exception as e:
        logger.error(f"Error fetching all users: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve users"}), 500


@app.route("/v1/admin/grievances", methods=["GET"])
@admin_required
def get_grievances():
    """Admin endpoint to get all grievance reports with patient names using a join."""
    try:
        # One query: join GrievanceReport -> User for patient name
        results = db.session.query(GrievanceReport, User.full_name)\
            .join(User, GrievanceReport.user_id == User.id)\
            .order_by(GrievanceReport.created_at.desc())\
            .all()

        grievances_data = []
        for report, patient_name in results:
            grievances_data.append({
                "id": report.report_id,
                "patient": patient_name or "Unknown",
                "subject": report.reason,
                "priority": report.priority,
                "status": report.status,
                "date": report.created_at.isoformat()
            })
        return jsonify({"success": True, "grievances": grievances_data})
    except Exception as e:
        logger.error(f"Error fetching grievances: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve grievances"}), 500


@app.route("/v1/doctor/dashboard", methods=["GET"])
def get_doctor_dashboard():
    """Doctor dashboard endpoint to get their specific appointments."""
    try:
        # Get doctor from session
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({"error": "Not authenticated as doctor"}), 401

        doctor = db.session.get(Doctor, doctor_id)
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        today_start = datetime.now().date()
        today_end = today_start + timedelta(days=1)

        appointments = Appointment.query.filter(
            Appointment.doctor_id == doctor.id,
            Appointment.appointment_datetime >= today_start,
            Appointment.appointment_datetime < today_end
        ).order_by(Appointment.appointment_datetime.asc()).all()

        appointments_data = []
        for appt in appointments:
            patient = User.query.get(appt.user_id)
            appointments_data.append({
                "id": appt.appointment_id,
                "patient": patient.full_name if patient else "Unknown",
                "time": appt.appointment_datetime.strftime('%I:%M %p'),
                "type": appt.appointment_type,
                "status": appt.status
            })

        return jsonify({
            "success": True,
            "doctorName": doctor.full_name,
            "appointments": appointments_data
        })
    except Exception as e:
        logger.error(f"Error fetching doctor dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load doctor dashboard"}), 500

@app.route("/v1/doctor/profile", methods=["GET"])
def get_doctor_profile():
    """Get doctor profile information"""
    try:
        doctor_id = session.get('doctor_id')
        if not doctor_id:
            return jsonify({"error": "Not authenticated as doctor"}), 401

        doctor = db.session.get(Doctor, doctor_id)
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        return jsonify({
            "success": True,
            "doctor": {
                "id": doctor.id,
                "doctorId": doctor.doctor_id,
                "fullName": doctor.full_name,
                "email": doctor.email,
                "specialization": doctor.specialization,
                "qualification": doctor.qualification,
                "experience": doctor.experience_years,
                "clinicName": doctor.clinic_name,
                "phoneNumber": doctor.phone_number,
                "profileImageUrl": doctor.profile_image_url
            }
        })
    except Exception as e:
        logger.error(f"Error fetching doctor profile: {e}", exc_info=True)
        return jsonify({"error": "Failed to load doctor profile"}), 500


@app.route("/v1/pharmacy/dashboard", methods=["GET"])
def get_pharmacy_dashboard():
    """Pharmacy dashboard endpoint to get orders and inventory alerts."""
    try:
        # Mock pharmacy ID
        pharmacy_id = 1

        pharmacy = Pharmacy.query.get(pharmacy_id)
        if not pharmacy:
            return jsonify({"error": "Pharmacy not found"}), 404

        new_orders = MedicineOrder.query.filter_by(
            pharmacy_id=pharmacy.id,
            status='Placed'
        ).order_by(MedicineOrder.created_at.desc()).all()

        orders_data = [
            {"id": order.order_id, "customer": User.query.get(order.user_id).full_name, "status": order.status}
            for order in new_orders
        ]

        # This would query an inventory table in a real app
        low_stock_alerts = 3

        return jsonify({
            "success": True,
            "pharmacyName": pharmacy.name,
            "newOrdersCount": len(orders_data),
            "pendingDeliveriesCount": MedicineOrder.query.filter_by(pharmacy_id=pharmacy.id, status='Out for Delivery').count(),
            "lowStockAlerts": low_stock_alerts,
            "orders": orders_data
        })
    except Exception as e:
        logger.error(f"Error fetching pharmacy dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load pharmacy dashboard"}), 500


# WebRTC Signaling Endpoints
@app.route("/v1/webrtc/offer", methods=["POST"])
def webrtc_offer():
    try:
        data = request.get_json() or {}
        appointment_id = data.get("appointmentId")
        offer_data = data.get("data")

        if not appointment_id or not offer_data:
            return jsonify({"error": "appointmentId and data required"}), 400

        with webrtc_lock:
            if appointment_id not in webrtc_signaling:
                webrtc_signaling[appointment_id] = {'offers': [], 'answers': [], 'ice': []}
            webrtc_signaling[appointment_id]['offers'].append(offer_data)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"WebRTC offer error: {e}")
        return jsonify({"error": "Failed to handle offer"}), 500

@app.route("/v1/webrtc/answer", methods=["POST"])
def webrtc_answer():
    try:
        data = request.get_json() or {}
        appointment_id = data.get("appointmentId")
        answer_data = data.get("data")

        if not appointment_id or not answer_data:
            return jsonify({"error": "appointmentId and data required"}), 400

        with webrtc_lock:
            if appointment_id not in webrtc_signaling:
                webrtc_signaling[appointment_id] = {'offers': [], 'answers': [], 'ice': []}
            webrtc_signaling[appointment_id]['answers'].append(answer_data)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"WebRTC answer error: {e}")
        return jsonify({"error": "Failed to handle answer"}), 500

@app.route("/v1/webrtc/ice", methods=["POST"])
def webrtc_ice():
    try:
        data = request.get_json() or {}
        appointment_id = data.get("appointmentId")
        ice_data = data.get("data")

        if not appointment_id or not ice_data:
            return jsonify({"error": "appointmentId and data required"}), 400

        with webrtc_lock:
            if appointment_id not in webrtc_signaling:
                webrtc_signaling[appointment_id] = {'offers': [], 'answers': [], 'ice': []}
            webrtc_signaling[appointment_id]['ice'].append(ice_data)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"WebRTC ICE error: {e}")
        return jsonify({"error": "Failed to handle ICE"}), 500

@app.route("/v1/webrtc/poll", methods=["GET"])
def webrtc_poll():
    try:
        appointment_id = request.args.get("appointmentId")
        if not appointment_id:
            return jsonify({"error": "appointmentId required"}), 400

        messages = []
        with webrtc_lock:
            if appointment_id in webrtc_signaling:
                room = webrtc_signaling[appointment_id]
                # Return all pending messages and clear them
                for offer in room['offers']:
                    messages.append({'type': 'offer', 'data': offer})
                for answer in room['answers']:
                    messages.append({'type': 'answer', 'data': answer})
                for ice in room['ice']:
                    messages.append({'type': 'ice', 'data': ice})
                # Clear after sending
                room['offers'] = []
                room['answers'] = []
                room['ice'] = []

        return jsonify(messages)
    except Exception as e:
        logger.error(f"WebRTC poll error: {e}")
        return jsonify({"error": "Failed to poll messages"}), 500

# Scheduled tasks and cleanup
def cleanup_on_exit():
    """Cleanup tasks on application shutdown"""
    logger.info("🔄 Application shutdown - performing cleanup...")
    try:
        with app.app_context():
            # Save all models
            save_all_models()

            # Update final metrics
            track_system_metrics()

            # Close database connections
            db.session.close()

        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

# Periodic tasks (every 24 hours)
def run_periodic_tasks():
    """Run periodic maintenance tasks"""
    try:
        logger.info("🔄 Running periodic maintenance tasks...")

        # Track daily metrics
        track_system_metrics()

        # Clean up old data (keep 90 days)
        if conversation_memory:
            conversation_memory.cleanup_old_data(days_to_keep=90)

        # Save models
        save_all_models()

        logger.info("✅ Periodic maintenance completed")
    except Exception as e:
        logger.error(f"❌ Error in periodic tasks: {e}")

def schedule_periodic_tasks():
    """Schedule periodic tasks to run every 24 hours"""
    def task_scheduler():
        while True:
            time.sleep(24 * 60 * 60)  # 24 hours
            run_periodic_tasks()

    scheduler_thread = threading.Thread(target=task_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Periodic task scheduler started")


# Root handler to satisfy uptime probes and browser requests
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify({
        "status": "ok",
        "service": "Sehat Sahara API",
        "version": "2.1.0",
        "endpoints_base": "/v1"
    }), 200
# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested resource does not exist",
        "available_endpoints": [
            "/v1/health", "/v1/register", "/v1/login", "/v1/logout",
            "/v1/predict", "/v1/book-doctor",
            "/v1/history", "/v1/user-stats", "/v1/ollama/status", "/v1/ollama/test",
            "/v1/dashboard", "/v1/doctors", "/v1/appointments", "/v1/appointments/cancel",
            "/v1/pharmacies", "/v1/medicines", "/v1/orders",
            "/v1/admin/users", "/v1/admin/grievances",
            "/v1/doctor/dashboard", "/v1/pharmacy/dashboard",
            "/v1/webrtc/offer", "/v1/webrtc/answer", "/v1/webrtc/ice", "/v1/webrtc/poll"
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The requested HTTP method is not supported for this endpoint"
    }), 405

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Request too large",
        "message": "The request payload is too large"
    }), 413

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "support_resources": {
            "emergency_services": "108",
            "health_helpline": "104"
        }
    }), 500



# Enhanced startup display with Ollama information
def display_startup_info():
    print("=" * 100)
    print("SEHAT SAHARA HEALTH ASSISTANT - Task-Oriented App Navigator")
    print("=" * 100)
    print()
    print("FEATURES:")
    print(" * Action JSON responses for app navigation")
    print(" * Emergency handling with TRIGGER_SOS (108)")
    print(" * Appointment booking, health records, pharmacy guidance")
    print()
    print("API ENDPOINTS:")
    print(" * POST /v1/register")
    print(" * POST /v1/login")
    print(" * POST /v1/logout")
    print(" * POST /v1/predict  (returns {response, action, parameters})")
    print(" * POST /v1/book-doctor")
    print(" * POST /v1/history")
    print(" * POST /v1/user-stats")
    print(" * GET  /v1/health")
    print(" * GET  /v1/ollama/status")
    print(" * POST /v1/ollama/test")
    print(" * GET  /v1/dashboard")
    print(" * GET  /v1/doctors")
    print(" * GET  /v1/appointments")
    print(" * POST /v1/appointments/cancel")
    print(" * GET  /v1/pharmacies")
    print(" * GET  /v1/medicines")
    print(" * POST /v1/orders")
    print(" * GET  /v1/admin/users")
    print(" * GET  /v1/admin/grievances")
    print(" * GET  /v1/doctor/dashboard")
    print(" * GET  /v1/pharmacy/dashboard")
    print()
    print("=" * 100)
    print("SYSTEM READY - SEHAT SAHARA ACTIVE")
    print("=" * 100)

if __name__ == "__main__":
    # Display startup information
    display_startup_info()

    # Initialize periodic tasks
    schedule_periodic_tasks()

    # Track initial system startup
    with app.app_context():
        try:
            track_system_metrics()
        except Exception as e:
            logger.error(f"Failed to track startup metrics: {e}")

    # Start the Flask application
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)