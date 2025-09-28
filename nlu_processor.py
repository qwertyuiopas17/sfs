"""
Sehat Sahara Health Assistant NLU Processor
Natural Language Understanding for Health App Navigation and Task-Oriented Commands
Supports Punjabi, Hindi, and English for rural patients
"""

import os
import pickle
import logging
import re
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import threading

# Try to import advanced NLP libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from api_ollama_integration import ollama_llama3
    HAS_OLLAMA = ollama_llama3.is_available if ollama_llama3 else False
except ImportError:
    HAS_OLLAMA = False
    ollama_llama3 = None

class ProgressiveNLUProcessor:
    """
    NLU processor for Sehat Sahara Health Assistant with multilingual support.
    Processes user commands for health app navigation and task completion.
    """

    def __init__(self, model_path: str = None, ollama_model: str = "phi"):
        self.logger = logging.getLogger(__name__)
        self.ollama_model = ollama_model
        self.use_ollama = False
        self._lock = threading.RLock()
        
        # Try to connect to API service
        if HAS_OLLAMA and ollama_llama3:
            try:
                if ollama_llama3.is_available:
                    self.use_ollama = True
                    self.logger.info(f"✅ API connection successful. Using API model for NLU processing.")
                else:
                    self.use_ollama = False
                    self.logger.warning(f"⚠️ API service not available. Falling back to keyword-based NLU.")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not connect to API service. Falling back to keyword-based NLU. Error: {e}")
                self.use_ollama = False

        # Initialize semantic model for enhanced understanding (optional)
        self.sentence_model = None
        self.use_semantic = False
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_semantic = True
                self.logger.info("✅ Semantic model loaded for enhanced NLU")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}")

        # Health app intent categories with multilingual keywords
        self.intent_categories = {
            'appointment_booking': {
                'keywords': [
                    # English
                    'book appointment', 'need to see doctor', 'doctor appointment', 'schedule appointment',
                    'meet doctor', 'consultation', 'book doctor', 'see doctor', 'doctor visit',
                    # Hindi (Latin script)
                    'doctor se milna hai', 'appointment book karni hai', 'doctor ko dikhana hai',
                    'doctor ke paas jana hai', 'appointment chahiye', 'doctor se baat karni hai',
                    # Punjabi (Latin script)
                    'doctor nu milna hai', 'appointment book karni hai', 'doctor kol jana hai',
                    'doctor nu dikhana hai', 'doctor de kol appointment', 'vaid nu milna hai'
                ],
                'urgency_indicators': ['urgent', 'emergency', 'turant', 'jaldi', 'emergency hai']
            },
            'appointment_view': {
                'keywords': [
                    # English
                    'my appointments', 'when is my appointment', 'next appointment', 'appointment time',
                    'show appointments', 'check appointment', 'appointment details',
                    # Hindi (Latin script)
                    'meri appointment kab hai', 'appointment ka time', 'appointment dekhni hai',
                    'kab hai appointment', 'appointment ki jankari',
                    # Punjabi (Latin script)
                    'meri appointment kado hai', 'appointment kado hai', 'appointment dekhan hai',
                    'appointment da time', 'appointment di jankari'
                ]
            },
            'appointment_cancel': {
                'keywords': [
                    # English
                    'cancel appointment', 'cancel my appointment', 'dont want appointment',
                    'remove appointment', 'delete appointment',
                    # Hindi (Latin script)
                    'appointment cancel karni hai', 'appointment nahi chahiye', 'appointment cancel karo',
                    'appointment hatana hai',
                    # Punjabi (Latin script)
                    'appointment cancel karni hai', 'appointment nahi chahidi', 'appointment cancel karo',
                    'appointment hatana hai'
                ]
            },
            'health_record_request': {
                'keywords': [
                    # English
                    'my reports', 'blood report', 'test results', 'medical records', 'health records',
                    'last report', 'show my reports', 'medical history', 'prescription history',
                    # Hindi (Latin script)
                    'meri report', 'blood report', 'test ka result', 'medical record',
                    'pichli report', 'dawai ki history', 'report dikhao',
                    # Punjabi (Latin script)
                    'meri report', 'blood report', 'test da result', 'medical record',
                    'pichli report', 'dawai di history', 'report dikhao'
                ]
            },
            'symptom_triage': {
                'keywords': [
                    # English
                    'fever', 'headache', 'pain', 'cough', 'cold', 'stomach pain', 'chest pain',
                    'feeling sick', 'not feeling well', 'symptoms', 'body ache',
                    # Hindi (Latin script)
                    'bukhar hai', 'sir dard hai', 'dard hai', 'khansi hai', 'pet dard hai',
                    'tabiyat kharab hai', 'bimari hai', 'body pain hai',
                    # Punjabi (Latin script)
                    'bukhar hai', 'sir dukh raha hai', 'dard hai', 'khansi hai', 'pet dukh raha hai',
                    'tabiyat kharab hai', 'bimari hai', 'body pain hai'
                ],
                'urgency_indicators': ['severe pain', 'chest pain', 'breathing problem', 'emergency', 'accident']
            },
            'find_medicine': {
                'keywords': [
                    # English
                    'find medicine', 'where to buy medicine', 'pharmacy near me', 'medicine shop',
                    'buy medicine', 'medicine available', 'find pharmacy',
                    # Hindi (Latin script)
                    'dawai kahan milegi', 'medicine shop', 'pharmacy', 'dawai leni hai',
                    'medicine kahan hai', 'dawai ki dukan',
                    # Punjabi (Latin script)
                    'dawai kithe milegi', 'medicine shop', 'pharmacy', 'dawai leni hai',
                    'medicine kithe hai', 'dawai di dukan'
                ]
            },
            'prescription_inquiry': {
                'keywords': [
                    # English
                    'how to take medicine', 'medicine dosage', 'when to take', 'medicine instructions',
                    'tablet kitni', 'medicine timing', 'prescription details',
                    # Hindi (Latin script)
                    'dawai kaise leni hai', 'kitni tablet leni hai', 'dawai ka time',
                    'medicine kab leni hai', 'dawai ki jankari',
                    # Punjabi (Latin script)
                    'dawai kive leni hai', 'kinni tablet leni hai', 'dawai da time',
                    'medicine kado leni hai', 'dawai di jankari'
                ]
            },
            'medicine_scan': {
                'keywords': [
                    # English
                    'scan medicine', 'check medicine', 'medicine scanner', 'identify medicine',
                    'what is this medicine', 'medicine name',
                    # Hindi (Latin script)
                    'medicine scan karo', 'ye kya dawai hai', 'medicine check karo',
                    'dawai ka naam', 'medicine identify karo',
                    # Punjabi (Latin script)
                    'medicine scan karo', 'eh ki dawai hai', 'medicine check karo',
                    'dawai da naam', 'medicine identify karo'
                ]
            },
            'emergency_assistance': {
                'keywords': [
                    # English
                    'emergency', 'help me', 'accident', 'urgent help', 'ambulance',
                    'emergency call', 'immediate help', 'crisis',
                    # Hindi (Latin script)
                    'emergency hai', 'help karo', 'accident hua hai', 'ambulance chahiye',
                    'turant help chahiye', 'emergency call',
                    # Punjabi (Latin script)
                    'emergency hai', 'help karo', 'accident ho gaya hai', 'ambulance chahida',
                    'turant help chahidi', 'emergency call'
                ],
                'urgency_indicators': ['emergency', 'accident', 'ambulance', 'urgent', 'help']
            },
            'report_issue': {
                'keywords': [
                    # English
                    'complaint', 'doctor was rude', 'overcharged', 'bad service', 'report problem',
                    'feedback', 'issue with', 'problem with',
                    # Hindi (Latin script)
                    'complaint hai', 'doctor rude tha', 'zyada paisa liya', 'service kharab thi',
                    'problem hai', 'shikayat hai',
                    # Punjabi (Latin script)
                    'complaint hai', 'doctor rude si', 'zyada paisa liya', 'service kharab si',
                    'problem hai', 'shikayat hai'
                ]
            },
            'general_inquiry': {
                'keywords': [
                    # English
                    'how to use app', 'help', 'what can you do', 'app features',
                    'how does this work', 'guide me', 'tutorial',
                    # Hindi (Latin script)
                    'app kaise use kare', 'help chahiye', 'app ki features',
                    'kaise kaam karta hai', 'guide karo',
                    # Punjabi (Latin script)
                    'app kive use karna hai', 'help chahidi', 'app dian features',
                    'kive kaam karda hai', 'guide karo'
                ]
            },
            'out_of_scope': {
                'keywords': [
                    # English
                    'weather', 'news', 'sports', 'movies', 'music', 'jokes', 'games',
                    'what is 10+10', 'tell me a story', 'sing a song',
                    # Hindi (Latin script)
                    'mausam kaisa hai', 'news kya hai', 'joke sunao', 'gaana gao',
                    'kahani sunao', 'khel',
                    # Punjabi (Latin script)
                    'mausam kaida hai', 'news ki hai', 'joke sunao', 'gana gao',
                    'kahani sunao', 'khel'
                ]
            }
        }

        self.conversation_stages = [
            'initial_contact', 'understanding', 'task_execution', 'confirmation',
            'completion', 'emergency_handling'
        ]

        # Build semantic embeddings if available
        if self.use_semantic:
            self._build_semantic_embeddings()

        # Load saved model if available
        if model_path and os.path.exists(model_path):
            self.load_nlu_model(model_path)

    def _build_semantic_embeddings(self):
        """Build semantic embeddings for each intent category"""
        try:
            self.category_embeddings = {}
            for category, data in self.intent_categories.items():
                # Use keywords to create pseudo-sentences for embedding
                keywords = data['keywords'][:5]  # Use top 5 keywords
                pseudo_sentences = [f"I want to {keyword}" for keyword in keywords]
                
                # Create embeddings
                embeddings = self.sentence_model.encode(pseudo_sentences)
                
                # Use mean embedding as category representation
                self.category_embeddings[category] = np.mean(embeddings, axis=0)
            
            self.logger.info("✅ Semantic embeddings built for all intent categories")
        except Exception as e:
            self.logger.error(f"Failed to build semantic embeddings: {e}")
            self.use_semantic = False

    def understand_user_intent(self, user_message: str, conversation_history: List[Dict[str, Any]] = None, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """
        Processes a user's message to understand intent and urgency for health app navigation.
        """
        cleaned_message = self._clean_and_preprocess(user_message)
        
        # Immediate check for out of scope content
        if self._is_out_of_scope(cleaned_message):
            return self._generate_out_of_scope_response()

        # Attempt to use API service for primary analysis
        if self.use_ollama:
            ollama_result = self._get_ollama_analysis(cleaned_message, conversation_history)
            if ollama_result:
                return self._compile_final_analysis(ollama_result, cleaned_message)

        # Fallback to keyword-based system
        self.logger.info(f"Using keyword-based NLU for message: '{cleaned_message[:50]}...'")
        fallback_result = self._get_fallback_analysis(cleaned_message, excluded_intents)
        return self._compile_final_analysis(fallback_result, cleaned_message)

    def _get_ollama_analysis(self, user_message: str, conversation_history: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Uses API service with conversation history for contextual NLU analysis."""
        try:
            history_str = "No previous conversation history."
            if conversation_history:
                history_context = []
                for turn in conversation_history:
                    role = turn.get('role', 'user').title()
                    content = turn.get('content', '')
                    history_context.append(f"{role}: {content}")
                history_str = "\n".join(history_context)

            prompt = f"""
You are the Sehat Sahara Health Assistant NLU system. Analyze the user's LATEST message for health app navigation intent. Return ONLY a valid JSON response:

{{
    "primary_intent": "one of: {', '.join(self.intent_categories.keys())}",
    "confidence": 0.85,
    "urgency_level": "low/medium/high/emergency",
    "language_detected": "en/hi/pa",
    "context_entities": {{ "doctor_type": "", "symptom": "", "medicine_name": "" }},
    "user_needs": ["app_navigation", "information", "booking"],
    "in_scope": true
}}

Guidelines:
- emergency_assistance: Only for explicit emergencies, accidents, or urgent medical help
- urgency_level: "emergency" only for life-threatening situations
- Use conversation HISTORY to understand context of short messages
- Detect language: en=English, hi=Hindi, pa=Punjabi
- Return ONLY valid JSON, no other text

HISTORY:
{history_str}

Analyze this message: '{user_message}'
"""

            response_text = ollama_llama3.client.generate_response(prompt, max_tokens=300, temperature=0.3)
            if not response_text:
                return None

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                self.logger.info(f"✅ API NLU analysis successful for: {user_message[:50]}...")
                return analysis
            else:
                return None

        except Exception as e:
            self.logger.error(f"❌ API NLU analysis failed: {e}")
            return None

    def _get_fallback_analysis(self, message: str, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """Generates NLU analysis using keywords for health app navigation."""
        
        # Handle short, context-dependent messages
        if len(message.split()) <= 2:
            self.logger.info(f"Short message detected: '{message}'. Using general_inquiry intent.")
            return {
                'primary_intent': 'general_inquiry',
                'confidence': 0.7,
                'urgency_level': 'low',
                'language_detected': self._detect_language(message),
                'context_entities': {},
                'user_needs': ['guidance'],
                'in_scope': True
            }

        # Perform keyword-based analysis
        analysis = self._comprehensive_intent_detection(message, excluded_intents)
        urgency_analysis = self._assess_urgency_and_severity(message, analysis)
        context_entities = self._extract_health_context(message)
        language_detected = self._detect_language(message)
        user_needs = self._identify_user_needs(analysis['primary_intent'])

        return {
            'primary_intent': analysis['primary_intent'],
            'confidence': analysis['confidence'],
            'urgency_level': urgency_analysis['urgency_level'],
            'language_detected': language_detected,
            'context_entities': context_entities,
            'user_needs': user_needs,
            'in_scope': True
        }

    def _comprehensive_intent_detection(self, message: str, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """Combines keyword matching for health app intent detection."""
        keyword_scores = self._enhanced_keyword_intent_detection(message)
        
        if excluded_intents:
            for intent in excluded_intents:
                if intent in keyword_scores:
                    keyword_scores[intent] *= 0.1

        if not keyword_scores:
            primary_intent = 'general_inquiry'
            confidence = 0.3
        else:
            primary_intent = max(keyword_scores, key=keyword_scores.get)
            confidence = keyword_scores[primary_intent]

        return {
            'primary_intent': primary_intent,
            'confidence': min(confidence, 1.0),
            'all_scores': keyword_scores
        }

    def _enhanced_keyword_intent_detection(self, message: str) -> Dict[str, float]:
        """Detects intent based on keywords with multilingual support."""
        scores = {}
        for category, data in self.intent_categories.items():
            score = 0.0
            for keyword in data['keywords']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', message, re.IGNORECASE):
                    score += 0.2 * len(keyword.split())  # Weight longer phrases more

            # Boost score for urgency indicators
            for urgency_indicator in data.get('urgency_indicators', []):
                if re.search(r'\b' + re.escape(urgency_indicator) + r'\b', message, re.IGNORECASE):
                    score *= 1.5

            if score > 0:
                scores[category] = min(score, 1.0)
        return scores

    def _assess_urgency_and_severity(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Assesses urgency based on keywords and intent for health app context."""
        intent = analysis['primary_intent']
        
        # Emergency keywords
        emergency_keywords = ['emergency', 'accident', 'ambulance', 'help me', 'urgent help', 
                             'emergency hai', 'accident hua hai', 'turant help', 'emergency call']
        
        # High urgency symptoms
        urgent_symptoms = ['chest pain', 'breathing problem', 'severe pain', 'unconscious',
                          'chest mein dard', 'saans nahi aa rahi', 'behosh']
        
        urgency_level = 'low'
        
        if intent == 'emergency_assistance' or any(keyword in message.lower() for keyword in emergency_keywords):
            urgency_level = 'emergency'
        elif any(symptom in message.lower() for symptom in urgent_symptoms):
            urgency_level = 'high'
        elif intent == 'symptom_triage':
            urgency_level = 'medium'

        return {
            'urgency_level': urgency_level
        }

    def _extract_health_context(self, message: str) -> Dict[str, str]:
        """Extracts health-related entities from the message."""
        context = {}
        
        # Doctor specialties
        specialties = ['cardiologist', 'dermatologist', 'pediatrician', 'gynecologist', 
                      'orthopedic', 'neurologist', 'heart doctor', 'skin doctor', 'child doctor']
        for specialty in specialties:
            if specialty in message.lower():
                context['doctor_type'] = specialty
                break
        
        # Common symptoms
        symptoms = ['fever', 'headache', 'cough', 'pain', 'cold', 'bukhar', 'sir dard', 'khansi']
        for symptom in symptoms:
            if symptom in message.lower():
                context['symptom'] = symptom
                break
        
        return context

    def _detect_language(self, message: str) -> str:
        """Simple language detection based on script and common words."""
        # Hindi indicators
        hindi_words = ['hai', 'kya', 'kaise', 'kab', 'kahan', 'meri', 'mera', 'chahiye', 'leni', 'dard']
        # Punjabi indicators  
        punjabi_words = ['hai', 'ki', 'kive', 'kado', 'kithe', 'meri', 'mera', 'chahidi', 'leni', 'dukh']
        
        message_lower = message.lower()
        
        hindi_count = sum(1 for word in hindi_words if word in message_lower)
        punjabi_count = sum(1 for word in punjabi_words if word in message_lower)
        
        if hindi_count > punjabi_count and hindi_count > 0:
            return 'hi'
        elif punjabi_count > 0:
            return 'pa'
        else:
            return 'en'

    def _identify_user_needs(self, primary_intent: str) -> List[str]:
        """Identifies user needs based on intent."""
        need_mapping = {
            'appointment_booking': ['booking', 'doctor_connection'],
            'appointment_view': ['information', 'schedule_check'],
            'appointment_cancel': ['booking_management'],
            'health_record_request': ['information', 'record_access'],
            'symptom_triage': ['health_assessment', 'guidance'],
            'find_medicine': ['pharmacy_search', 'medicine_availability'],
            'prescription_inquiry': ['information', 'medicine_guidance'],
            'medicine_scan': ['medicine_identification'],
            'emergency_assistance': ['immediate_help', 'emergency_services'],
            'report_issue': ['feedback', 'complaint_handling'],
            'general_inquiry': ['guidance', 'app_navigation'],
            'out_of_scope': ['redirection']
        }
        return need_mapping.get(primary_intent, ['guidance'])

    def _is_out_of_scope(self, message: str) -> bool:
        """Check if message is out of scope for health app."""
        out_of_scope_keywords = self.intent_categories['out_of_scope']['keywords']
        return any(keyword in message.lower() for keyword in out_of_scope_keywords)

    def _generate_out_of_scope_response(self) -> Dict[str, Any]:
        """Returns structured response for out of scope content."""
        return {
            'primary_intent': 'out_of_scope',
            'confidence': 0.95,
            'urgency_level': 'low',
            'language_detected': 'en',
            'context_entities': {},
            'user_needs': ['redirection'],
            'in_scope': False
        }

    def _clean_and_preprocess(self, message: str) -> str:
        """Cleans and standardizes the user's message for analysis."""
        cleaned = message.lower().strip()
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not", "didn't": "did not",
            "i'm": "i am", "you're": "you are", "it's": "it is", "i've": "i have"
        }

        for contraction, expansion in contractions.items():
            cleaned = cleaned.replace(contraction, expansion)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return cleaned

    def _compile_final_analysis(self, analysis_data: Dict[str, Any], cleaned_message: str) -> Dict[str, Any]:
        """Compiles the final NLU response object from the analysis data."""
        primary_intent_value = analysis_data.get('primary_intent', 'general_inquiry')
        if isinstance(primary_intent_value, list) and len(primary_intent_value) > 0:
            primary_intent_value = primary_intent_value[0]
        elif not isinstance(primary_intent_value, str):
            primary_intent_value = 'general_inquiry'

        conversation_stage = self._determine_conversation_stage(cleaned_message, {'primary_intent': primary_intent_value})

        return {
            'primary_intent': primary_intent_value,
            'confidence': float(analysis_data.get('confidence', 0.5)),
            'urgency_level': analysis_data.get('urgency_level', 'low'),
            'language_detected': analysis_data.get('language_detected', 'en'),
            'context_entities': analysis_data.get('context_entities', {}),
            'conversation_stage': conversation_stage,
            'user_needs': analysis_data.get('user_needs', ['guidance']),
            'in_scope': bool(analysis_data.get('in_scope', True)),
            'processing_timestamp': datetime.now().isoformat(),
            'api_analysis_used': self.use_ollama
        }

    def _determine_conversation_stage(self, message: str, analysis: Dict) -> str:
        """Determines the current stage of the conversation for health app context."""
        intent = analysis['primary_intent']
        
        if intent == 'emergency_assistance':
            return 'emergency_handling'
        elif intent in ['appointment_booking', 'find_medicine', 'medicine_scan']:
            return 'task_execution'
        elif intent in ['appointment_view', 'health_record_request', 'prescription_inquiry']:
            return 'information_retrieval'
        else:
            return 'understanding'

    # Backward compatibility and utility methods
    def save_nlu_model(self, filepath: str) -> bool:
        """Save NLU model configuration and learned parameters."""
        try:
            with self._lock:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                config = {
                    'intent_categories': self.intent_categories,
                    'conversation_stages': self.conversation_stages,
                    'use_ollama': self.use_ollama,
                    'ollama_model': self.ollama_model,
                    'use_semantic': self.use_semantic,
                    'model_version': '3.0.0',
                    'save_timestamp': datetime.now().isoformat()
                }
                
                if hasattr(self, 'category_embeddings') and self.category_embeddings:
                    config['category_embeddings'] = {
                        category: embedding.tolist() 
                        for category, embedding in self.category_embeddings.items()
                    }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(config, f)
                
                self.logger.info(f"✅ NLU model configuration saved to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save NLU model: {e}")
            return False

    def load_nlu_model(self, filepath: str) -> bool:
        """Load NLU model configuration and parameters."""
        try:
            with self._lock:
                with open(filepath, 'rb') as f:
                    config = pickle.load(f)
                
                self.intent_categories = config.get('intent_categories', self.intent_categories)
                self.conversation_stages = config.get('conversation_stages', self.conversation_stages)
                self.ollama_model = config.get('ollama_model', self.ollama_model)
                
                if 'category_embeddings' in config and self.use_semantic:
                    self.category_embeddings = {
                        category: np.array(embedding) 
                        for category, embedding in config['category_embeddings'].items()
                    }
                
                self.logger.info(f"✅ NLU model configuration loaded from {filepath}")
                return True
                
        except FileNotFoundError:
            self.logger.warning(f"⚠️ NLU model file not found: {filepath}. Using defaults.")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading NLU model: {e}. Using defaults.")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'model_type': 'Sehat Sahara Health Assistant NLU Processor',
            'version': '3.0.0',
            'api_enabled': self.use_ollama,
            'api_model': self.ollama_model if self.use_ollama else None,
            'semantic_enabled': self.use_semantic,
            'intent_categories_count': len(self.intent_categories),
            'conversation_stages_count': len(self.conversation_stages),
            'supported_languages': ['English', 'Hindi', 'Punjabi'],
            'initialized_at': datetime.now().isoformat()
        }

    def validate_configuration(self) -> bool:
        """Validate the current model configuration."""
        try:
            if not self.intent_categories:
                self.logger.error("❌ No intent categories defined")
                return False
            
            if not self.conversation_stages:
                self.logger.error("❌ No conversation stages defined")
                return False
            
            test_result = self.understand_user_intent("book appointment")
            if not test_result or 'primary_intent' not in test_result:
                self.logger.error("❌ Basic intent detection failed")
                return False
            
            self.logger.info("✅ NLU model configuration is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    # Backward compatibility methods
    def analyze_user_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility method - alias for understand_user_intent."""
        return self.understand_user_intent(message, context.get('excluded_intents') if context else None)

    def get_intent_confidence(self, message: str, intent: str) -> float:
        """Get confidence score for a specific intent."""
        result = self.understand_user_intent(message)
        return result.get('confidence', 0.0) if result.get('primary_intent') == intent else 0.0

    def is_emergency_detected(self, message: str) -> bool:
        """Quick check if message indicates emergency situation."""
        result = self.understand_user_intent(message)
        return result['urgency_level'] == 'emergency' or result['primary_intent'] == 'emergency_assistance'