# Complete Enhanced Mental Health Chatbot Application - Enhanced with Ollama Llama 3
# This is the COMPLETE version maintaining ALL original functionality

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
    db, User, Doctor, Appointment, HealthRecord, Pharmacy,
    ConversationTurn, UserSession, init_database, get_user_statistics
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
CORS(app, supports_credentials=True, resources={
    r"/v1/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
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
        logger.info("🧠 Initializing Progressive Conversation Memory...")
        from conversation_memory import ProgressiveConversationMemory  # keep memory
        global conversation_memory
        conversation_memory = ProgressiveConversationMemory(save_path=memory_model_path)
        system_status['conversation_memory'] = True
        logger.info("✅ Conversation Memory initialized successfully")
        
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
        logger.info("✅ Database initialized successfully")
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
            logger.info("✅ NLU model saved")
        
        if conversation_memory and system_status['conversation_memory']:
            conversation_memory.save_memory()
            logger.info("✅ Conversation memory saved")
        
        # Crisis detector model saving removed as it's no longer used
        
        logger.info("✅ All models saved successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving models: {e}")
        return False

def track_system_metrics():
    """Track and update system-wide metrics"""
    try:
        today = datetime.now().date()
        
        # Check if metrics already exist for today
        existing_metrics = SystemMetrics.query.filter_by(metrics_date=today).first()
        
        if not existing_metrics:
            # Calculate metrics for today
            total_users = User.query.filter_by(is_active=True).count()
            new_users = User.query.filter(
                User.created_at >= datetime.combine(today, datetime.min.time())
            ).count()
            
            total_conversations = ConversationTurn.query.filter(
                ConversationTurn.timestamp >= datetime.combine(today, datetime.min.time())
            ).count()
            
            # Crisis events detection removed as it's no longer used
            
            methods_suggested = ConversationTurn.query.filter(
                ConversationTurn.timestamp >= datetime.combine(today, datetime.min.time()),
                ConversationTurn.method_suggested.isnot(None)
            ).count()
            
            counselor_referrals = CounselorInteraction.query.filter(
                CounselorInteraction.referral_date >= datetime.combine(today, datetime.min.time())
            ).count()
            
            # Calculate method success rate
            method_feedback = MethodFeedback.query.filter(
                MethodFeedback.feedback_date >= datetime.combine(today, datetime.min.time())
            ).all()
            
            if method_feedback:
                effective_methods = sum(1 for feedback in method_feedback 
                                     if feedback.effectiveness_rating == 'effective')
                method_success_rate = effective_methods / len(method_feedback)
            else:
                method_success_rate = 0.0
            
            # Create metrics record
            metrics = SystemMetrics(
                metrics_date=today,
                total_active_users=total_users,
                new_users_registered=new_users,
                total_conversations=total_conversations,
                # crisis_events_detected=crisis_events, # removed
                methods_suggested_total=methods_suggested,
                methods_marked_effective=sum(1 for feedback in method_feedback 
                                           if feedback.effectiveness_rating == 'effective'),
                overall_method_success_rate=method_success_rate,
                counselor_referrals_made=counselor_referrals
            )
            
            db.session.add(metrics)
            db.session.commit()
            logger.info(f"✅ System metrics updated for {today}")
            
    except Exception as e:
        logger.error(f"❌ Error tracking system metrics: {e}")

def get_current_user():
    """Security helper to get current authenticated user"""
    user_id = session.get('user_id')
    if user_id:
        try:
            return User.query.get(user_id)
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None
    return None

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
        patient_id = data.get("patientId", "").strip()
        village = data.get("village", "").strip()
        district = data.get("district", "").strip()
        state = data.get("state", "Punjab").strip()
        pincode = data.get("pincode", "").strip()
        preferred_language = data.get("preferredLanguage", "hi").strip()

        errors = []
        if not email or "@" not in email: errors.append("Valid email is required")
        if not password or len(password) < 8: errors.append("Password must be at least 8 characters")
        if not full_name: errors.append("Full name is required")
        if errors:
            return jsonify({"success": False, "message": "Fix the errors below.", "errors": errors}), 400

        # Uniqueness check on email or patient_id
        existing = User.query.filter(
            (User.email == email) | (User.patient_id == patient_id if patient_id else False)
        ).first()
        if existing:
            return jsonify({"success": False, "message": "Account with this email or patient ID already exists."}), 409

        # Auto-generate patient ID if missing
        if not patient_id:
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
            state=state or "Punjab",
            pincode=pincode,
            preferred_language=preferred_language
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        create_user_session(new_user, {
            'remote_addr': request.environ.get('REMOTE_ADDR'),
            'user_agent': request.environ.get('HTTP_USER_AGENT')
        })

        # Initialize conversation memory
        if conversation_memory:
            try:
                conversation_memory.create_or_get_user(patient_id)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "patientId": patient_id,
            "message": f"Welcome {full_name}! Your account has been created.",
            "user": {
                "patientId": patient_id,
                "fullName": full_name,
                "email": email,
                "phoneNumber": phone_number,
                "location": ", ".join([village, district, state, pincode]).strip(", "),
                "memberSince": new_user.created_at.isoformat(),
                "preferredLanguage": preferred_language
            }
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {e}")
        update_system_state('register', success=False)
        return jsonify({"success": False, "message": "Registration failed."}), 500

@app.route("/v1/login", methods=["POST"])
def login():
    try:
        update_system_state('login')
        data = request.get_json() or {}
        patient_id = data.get("patientId", "").strip()
        password = data.get("password", "")

        if not patient_id or not password:
            return jsonify({"success": False, "message": "Patient ID and password are required."}), 400

        user = User.query.filter_by(patient_id=patient_id, is_active=True).first()
        if user and user.check_password(password):
            user.update_last_login()
            db.session.commit()
            session.permanent = True
            session['user_id'] = user.id
            session['patient_id'] = user.patient_id
            session['login_time'] = datetime.now().isoformat()

            create_user_session(user, {
                'remote_addr': request.environ.get('REMOTE_ADDR'),
                'user_agent': request.environ.get('HTTP_USER_AGENT')
            })

            stats = get_user_statistics(user.id) or {}
            return jsonify({
                "success": True,
                "message": f"Welcome back, {user.full_name}!",
                "user": {
                    "patientId": user.patient_id,
                    "fullName": user.full_name,
                    "email": user.email,
                    "phoneNumber": user.phone_number,
                    "preferredLanguage": user.preferred_language,
                    "memberSince": user.created_at.isoformat(),
                    "lastLogin": user.last_login.isoformat() if user.last_login else None
                },
                "statistics": stats
            })

        update_system_state('login', success=False)
        return jsonify({"success": False, "message": "Invalid Patient ID or password."}), 401

    except Exception as e:
        logger.error(f"Login error: {e}")
        update_system_state('login', success=False)
        return jsonify({"success": False, "message": "Login failed due to server error."}), 500

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
        user_message = (data.get("message") or "").strip()
        image_b64 = data.get("imageData")
        user_id_str = (data.get("userId") or "").strip()
        context = data.get("context", {}) or {}
        
        if not user_id_str:
            return jsonify({"error": "User ID is required."}), 400
        if not user_message and not image_b64:
            return jsonify({"error": "Either message or imageData is required."}), 400
        
        # Load current user by patient_id
        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
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
        
        # Build short NLU history
        history_turns = conversation_memory.conversation_history.get(current_user.patient_id, [])
        nlu_history = []
        for turn in history_turns[-4:]:
            nlu_history.append({'role': 'user', 'content': turn.user_message, 'intent': getattr(turn, 'detected_intent', None)})
            nlu_history.append({'role': 'assistant', 'content': turn.bot_response, 'intent': getattr(turn, 'detected_intent', None)})

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
            # Use Sehat Sahara API client for structured JSON action
            action_payload = None
            if sehat_sahara_client and sehat_sahara_client.is_available:
                action_payload = sehat_sahara_client.generate_action_json(
                    user_message=effective_message,
                    nlu_result={**nlu_understanding, "language_detected": detected_language_code},
                    conversation_history=nlu_history,
                    language=detected_language_code
                )
            # Fallback to local response generator if API unavailable
            if not action_payload:
                local = response_generator.generate_response(
                    user_message=effective_message,
                    nlu_result={**nlu_understanding, "language_detected": detected_language_code},
                    user_context={"user_id": current_user.patient_id, "session_id": session.get('session_record_id')},
                    conversation_history=nlu_history
                )
                action_payload = local
        
        # Clean only the response text
        if action_payload.get("response"):
            action_payload["response"] = clean_ai_response(action_payload["response"])
        
        # Save conversation turn with action fields
        turn = ConversationTurn(
            user_id=current_user.id,
            user_message=effective_message,
            bot_response=action_payload.get("response", ""),
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
        
        # Increment appointments_booked counter if action booked appointment
        if action_payload.get("action") == "NAVIGATE_TO_APPOINTMENT_BOOKING":
            update_system_state('predict', appointments_booked=0)  # not actually booked yet; count on booking
        
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
        doctor_id_str = (data.get("doctorId") or "").strip()
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
def get_system_metrics():
    """Get system-wide metrics and analytics"""
    try:
        # Get recent metrics from database
        recent_metrics = SystemMetrics.query.order_by(SystemMetrics.metrics_date.desc()).limit(30).all()
        
        metrics_data = []
        for metric in recent_metrics:
            metrics_data.append({
                "date": metric.metrics_date.isoformat(),
                "total_users": metric.total_active_users,
                "new_users": metric.new_users_registered,
                "conversations": metric.total_conversations,
                # "crisis_events": metric.crisis_events_detected, # removed
                "methods_suggested": metric.methods_suggested_total,
                "method_success_rate": metric.overall_method_success_rate,
                "counselor_referrals": metric.counselor_referrals_made
            })
        
        # Get system analytics from memory if available
        system_analytics = {}
        if conversation_memory:
            try:
                system_analytics = conversation_memory.get_system_analytics()
            except Exception as e:
                logger.error(f"Error getting system analytics: {e}")
        
        # Current system state
        current_state = {
            **system_state,
            'component_health': system_status,
            'current_time': datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "historical_metrics": metrics_data,
            "system_analytics": system_analytics,
            "current_state": current_state,
            "summary": {
                "total_metrics_days": len(metrics_data),
                "latest_date": metrics_data[0]["date"] if metrics_data else None,
                "system_health": "healthy" if all(system_status.values()) else "degraded"
            }
        })
    
    except Exception as e:
        logger.error(f"❌ Metrics retrieval error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Failed to retrieve system metrics",
            "message": "Check system logs for details"
        }), 500

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

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested resource does not exist",
        "available_endpoints": [
            "/v1/health", "/v1/register", "/v1/login", "/v1/logout",
            "/v1/predict", "/v1/book-doctor",
            "/v1/history", "/v1/user-stats", "/v1/ollama/status", "/v1/ollama/test"
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
    print("🚀 SEHAT SAHARA HEALTH ASSISTANT - Task-Oriented App Navigator")
    print("=" * 100)
    print()
    print("🌟 FEATURES:")
    print(" ✅ Action JSON responses for app navigation")
    print(" ✅ Emergency handling with TRIGGER_SOS (108)")
    print(" ✅ Appointment booking, health records, pharmacy guidance")
    print()
    print("🌐 API ENDPOINTS:")
    print(" • POST /v1/register")
    print(" • POST /v1/login")
    print(" • POST /v1/logout")
    print(" • POST /v1/predict  (returns {response, action, parameters})")
    print(" • POST /v1/book-doctor")
    print(" • POST /v1/history")
    print(" • POST /v1/user-stats")
    print(" • GET  /v1/health")
    print(" • GET  /v1/ollama/status")
    print(" • POST /v1/ollama/test")
    print()
    print("=" * 100)
    print("🎉 SYSTEM READY - SEHAT SAHARA ACTIVE")
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
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
