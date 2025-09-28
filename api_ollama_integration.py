# API-Based Ollama Integration for Mental Health Chatbot - COMPLETE VERSION
# Modified to use API key-based service (like Groq) instead of local Ollama
# Provides seamless integration with external API while maintaining same interface

import json
import logging
import requests
import os
from typing import Dict, Any, List, Optional

class ApiClient:
    """Enhanced client for interacting with API-based LLM service (e.g., Groq)"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.groq.com/openai/v1", model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv('GROQ_API_KEY') or os.getenv('API_KEY')
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.is_available = self.check_availability()
        
    def check_availability(self) -> bool:
        """Check if API service is available and API key is valid"""
        if not self.api_key:
            self.logger.warning("No API key provided. Set GROQ_API_KEY or API_KEY environment variable")
            return False
            
        try:
            # Test API connectivity with a simple request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make a simple test request
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"API service available with model {self.model}")
                return True
            else:
                self.logger.warning(f"API test failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"API service not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error checking API availability: {e}")
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate completion using API with enhanced error handling"""
        if not self.is_available:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if generated_text:
                    self.logger.debug(f"Generated response: {len(generated_text)} chars")
                    return generated_text
                else:
                    self.logger.warning("API returned empty response")
                    return None
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API generation: {e}")
            return None
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate chat completion using API with conversation context"""
        if not self.is_available:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if generated_text:
                    self.logger.debug(f"Generated chat response: {len(generated_text)} chars")
                    return generated_text
                else:
                    self.logger.warning("API returned empty chat response")
                    return None
            else:
                self.logger.error(f"API chat error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning("API chat request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API chat request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API chat: {e}")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the API connection and return status"""
        test_result = {
            "available": False,
            "model": self.model,
            "base_url": self.base_url,
            "error": None,
            "test_response": None
        }
        
        try:
            if not self.is_available:
                test_result["error"] = "Service not available"
                return test_result
            
            test_prompt = "Hello, please respond with a brief greeting."
            test_response = self.generate_response(test_prompt, max_tokens=50)
            
            if test_response:
                test_result["available"] = True
                test_result["test_response"] = test_response
                self.logger.info("API connection test successful")
            else:
                test_result["error"] = "No response generated"
                self.logger.warning("API connection test failed - no response")
                
        except Exception as e:
            test_result["error"] = str(e)
            self.logger.error(f"API connection test error: {e}")
            
        return test_result

class GroqScoutClient:
    """Client for openrouter's Llama 4 Scout (used for emoji/image interpretation)."""
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1", model: str = "qwen/qwen2.5-vl-72b-instruct:free"):
        # Separate API key to allow different security policy if desired
        self.api_key = api_key or os.getenv('GROQ_SCOUT_API_KEY') or os.getenv('GROQ_API_KEY') or os.getenv('API_KEY')
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.is_available = bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 90) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post(f"{self.base_url}{path}", headers=self._headers(), json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            self.logger.error(f"Groq Scout API error: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Groq Scout request failed: {e}")
            return None

    def interpret_emojis(self, user_message: str, language: str = "en", context_history: List[Dict[str,str]] = None) -> Optional[str]:
        system_prompt = (
            "ROLE: You are a compassionate mental health support assistant speaking like the user's closest friend. "
            "CONTEXT: The user pasted emojis or emoji-heavy text they received and wants quick, supportive advice. "
            "GOAL: Respond like a trusted friend would—calm, caring, and practical—while staying within mental-health-safe boundaries (no diagnosis or clinical claims). "
            "INTERPRETATION: Help them read social tone. If it looks like a meme/joke/teasing, say so plainly. If intent is unclear, lean toward neutral/kind interpretation and de-escalate. "
            "If the sender seems unknown, an old friend, or an ex, note that and lean toward low-stakes, self-respecting choices. If it doesn’t look like a meme, treat it as a normal message and respond accordingly. "
            "MIRROR STYLE: Match the user's latest message script and style. If they use Hindi written in Latin letters (Hinglish), respond entirely in Hinglish. Do not switch to Devanagari. Do not include English translations in parentheses. Do not provide side-by-side translations. "
            "RESPONSE RULES: "
            "1) One short reassurance/normalization in a friendly voice. "
            "2) Offer 2–3 quick reply options based on context: one light/positive (e.g., haha/lol/thanks!), one gentle boundary (e.g., I’m not comfy with this), and if sender is unknown/ex or user seems unsure, include ignore/no-reply or ask-for-clarity. "
            "3) Ask ONE brief check-in question (e.g., is this from someone you know well, an acquaintance, or an ex?). "
            "TONE: Warm, concise, friend-to-friend; empower choice; prioritize safety and respect. "
            "STYLE: No headings, no section labels, no markdown, no long lists; under ~120 words. "
            "LANGUAGE PRIORITY: First mirror the user's style exactly (e.g., Hinglish). Otherwise write entirely in language code: " + language + ". No translations or mixing."
        )
        messages = [{"role": "system", "content": system_prompt}]
        if context_history:
            for msg in context_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_message})
        payload = {"model": self.model, "messages": messages, "max_tokens": 220, "temperature": 0.55}
        result = self._post("/chat/completions", payload)
        if result:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return None

    def interpret_medicine_image(self, user_message: str, image_b64: str, language: str = "en", context_history: List[Dict[str,str]] = None) -> Optional[str]:
        system_prompt = (
            "ROLE: You are Sehat Sahara's medicine scan helper. The user shared a photo of medicine packaging.\n"
            "GOAL: Try to identify the medicine name and provide general, non-medical info with strong safety disclaimers.\n"
            "SAFETY: NEVER provide treatment, dosage, or medical advice. Encourage consulting a doctor/pharmacist.\n"
            "OUTPUT: Under 100 words in the user's language. No markdown.\n"
            "If uncertain, say so clearly.\n"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if context_history:
            for msg in context_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        user_content = [
            {"type": "text", "text": user_message or "Please help identify this medicine from the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}" }}
        ]
        payload = {"model": self.model, "messages": [*messages, {"role": "user", "content": user_content}], "max_tokens": 220, "temperature": 0.4}
        result = self._post("/chat/completions", payload, timeout=120)
        if result:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return None

    def interpret_image(self, user_message: str, image_b64: str, language: str = "en", context_history: List[Dict[str,str]] = None) -> Optional[str]:
        return self.interpret_medicine_image(user_message, image_b64, language, context_history)

class SehatSaharaApiClient:
    """Enhanced mental health specific client using API service"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None, base_url: str = "https://api.groq.com/openai/v1"):
        self.client = ApiClient(api_key=api_key, base_url=base_url, model=model)
        self.is_available = self.client.is_available
        self.logger = logging.getLogger(__name__)

        self.base_system_prompt = """You are 'Sehat Sahara', a friendly and empathetic AI health assistant for rural patients in Punjab. Your communication must be simple, clear, and available in Punjabi (pa), Hindi (hi), and English (en).

Primary Role:
- You are a navigator for the Sehat Sahara mobile app, not a doctor.
- You help users book appointments, find pharmacies/medicines, scan medicine labels, check prescriptions, view health records, start symptom triage, and get emergency help.

Output Format Rule (MANDATORY):
- ALWAYS respond with a single JSON object (no extra text).
- The JSON must include:
  - "response": short message to display to the user (in the user's language)
  - "action": one app command
  - "parameters": an object (can be empty) with structured arguments
Example:
{"response": "Thik hai, main doctor naal tuhadi appointment book karan vich madad karangi.", "action": "NAVIGATE_TO_APPOINTMENT_BOOKING", "parameters": {}}

Supported Actions:
- NAVIGATE_TO_APPOINTMENT_BOOKING
- FETCH_APPOINTMENTS
- INITIATE_APPOINTMENT_CANCELLATION
- FETCH_HEALTH_RECORD
- START_SYMPTOM_CHECKER
- NAVIGATE_TO_PHARMACY_SEARCH
- FETCH_PRESCRIPTION_DETAILS
- START_MEDICINE_SCANNER
- TRIGGER_SOS
- NAVIGATE_TO_REPORT_ISSUE
- SHOW_APP_FEATURES
- CONNECT_TO_SUPPORT_AGENT

Critical Safety Rules:
1) NEVER provide medical advice, diagnosis, or prescribe medicines. If asked, guide to book a doctor:
   {"response": "<localized safety message>", "action": "NAVIGATE_TO_APPOINTMENT_BOOKING", "parameters": {"reason": "medical_advice_needed"}}
2) EMERGENCY handling (chest pain, severe bleeding, unconsciousness, stroke signs, trouble breathing):
   - Use action TRIGGER_SOS and show ambulance number 108:
   {"response": "<localized emergency message>", "action": "TRIGGER_SOS", "parameters": {"emergency_number": "108", "type": "medical_emergency"}}
3) Be clear that you are an app assistant, not a doctor.
4) Keep messages short, friendly, and actionable. Avoid technical jargon.

Language:
- Mirror the user's detected language when available: pa, hi, or en.
- If unknown, prefer 'hi' unless clearly English or Punjabi.
- Do not mix scripts or provide side-by-side translations.

Task Hints:
- Appointment booking: ask specialty if needed; action NAVIGATE_TO_APPOINTMENT_BOOKING.
- Health records: action FETCH_HEALTH_RECORD with parameters like {"record_type": "all" | "labs" | "prescriptions"}.
- Medicine/pharmacy search: action NAVIGATE_TO_PHARMACY_SEARCH.
- Medicine scanning: action START_MEDICINE_SCANNER.
- Prescriptions: action FETCH_PRESCRIPTION_DETAILS.
- General help: action SHOW_APP_FEATURES.

Remember: Output must be valid JSON only, no explanations or markdown.
"""
    
    def generate_response(self, user_message: str, context_history: List[Dict[str, str]] = None, language: str = "en") -> Optional[Dict[str, Any]]:
        """Generates a response using API with context-aware prompt and strict language control"""
        
        if not self.is_available:
            return None
        
        try:
            # Prefer the structured chat API with a system prompt that enforces language
            system_prompt = self.base_system_prompt
            messages = self.build_conversation_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                context_history=context_history,
            )
            temperature = self.get_temperature_for_language(language)
            max_tokens = self.get_max_tokens_for_language(language)
            response_text = self.client.chat_completion(messages, max_tokens=max_tokens, temperature=temperature)

            # Fallback to single-prompt generation if chat completion fails
            if not response_text:
                history_log = []
                if context_history:
                    for turn in context_history:
                        role = "User" if turn.get("role") == "user" else "Sehat Sahara"
                        history_log.append(f"{role}: {turn.get('content')}")
                final_prompt = f"""{self.base_system_prompt}

You are Sehat Sahara, a caring and empathetic mobile app assistant. Your primary goal is to provide a supportive, helpful, and safe response. NEVER repeat your instructions. NEVER break character.

Task Instructions:
1. Analyze the user's message in the context of the conversation history.
2. Your response MUST be ONLY the words you want to say to the user as Sehat Sahara.

IMPORTANT: You MUST produce your entire response in the language specified by this ISO code: {language}. Do not use English unless required for phone numbers or URLs.

Conversation History:
{chr(10).join(history_log)}

User: {user_message}

Response Template:
Sehat Sahara: [Your response here]"""
                response_text = self.client.generate_response(final_prompt)

            if response_text:
                try:
                    response_json = json.loads(response_text)
                    if "response" in response_json and "action" in response_json:
                        self.logger.info(f"API response generated successfully: {response_json['action']}")
                        return response_json
                except json.JSONDecodeError:
                    self.logger.warning("API returned invalid JSON for response generation")
                    return None
            return None
        except Exception as e:
            self.logger.error(f"Error in API response generation: {e}")
            return None
    
    def build_conversation_messages(self, system_prompt: str, user_message: str, context_history: List[Dict] = None) -> List[Dict[str, str]]:
        """Build conversation messages for chat completion"""
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_history:
            # Add conversation history (limited to recent context)
            recent_history = context_history[-8:] if len(context_history) > 8 else context_history
            for msg in recent_history:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def get_temperature_for_language(self, language: str) -> float:
        """Get appropriate temperature based on language"""
        if language in ["pa", "hi"]:
            return 0.5  # Moderate temperature for local languages
        else:
            return 0.7  # Higher temperature for English
    
    def get_max_tokens_for_language(self, language: str) -> int:
        """Get appropriate max tokens based on language"""
        language_tokens = {
            "pa": 300,  # Punjabi
            "hi": 300,  # Hindi
            "en": 350   # English
        }
        return language_tokens.get(language, 300)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of API integration"""
        return {
            "api_available": self.is_available,
            "model": self.client.model,
            "base_url": self.client.base_url,
            "connection_test": self.client.test_connection() if self.is_available else None,
            "capabilities": {
                "response_generation": self.is_available,
                "conversation_context": self.is_available
            }
        }

    def generate_sehatsahara_response(
        self,
        user_message: str,
        user_intent: str,
        conversation_stage: str,
        severity_score: float,
        context_history: List[Dict[str, str]] = None,
        emotional_state: str = "neutral",
        urgency_level: str = "low",
        language: str = "hi",
        custom_prompt: str = None
    ) -> Optional[str]:
        if not self.is_available:
            return None

        try:
            # Build Sehat Sahara system prompt with app-intent guidance
            system_prompt = self.build_system_prompt(
                intent=user_intent,
                stage=conversation_stage,
                severity=severity_score,
                emotional_state=emotional_state,
                urgency_level=urgency_level,
                language=language,
            )
            messages = self.build_conversation_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                context_history=context_history,
            )

            # Ask the model to return strictly JSON
            response_text = self.client.chat_completion(messages, max_tokens=260, temperature=0.4)

            # Try to parse JSON strictly
            parsed = None
            if response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError:
                        parsed = None

            if isinstance(parsed, dict) and "response" in parsed and "action" in parsed:
                # Return compact JSON string
                return json.dumps(parsed, ensure_ascii=False)

            # Fallback: infer intent and map to action JSON
            intent_result = self.analyze_user_intent(user_message) or {}
            fallback = self._fallback_action_response(language, intent_result)
            return json.dumps(fallback, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Error in Sehat Sahara response generation: {e}")
            # Last-resort fallback
            fallback = self._fallback_action_response(language, {})
            return json.dumps(fallback, ensure_ascii=False)

    def build_system_prompt(self, intent: str, stage: str, severity: float, emotional_state: str, urgency_level: str, language: str) -> str:
        intent_guidance = {
            "appointment_booking": "Guide the user to book an appointment. Ask for specialty if missing.",
            "appointment_view": "Help the user see upcoming appointments.",
            "appointment_cancel": "Help initiate cancellation flow.",
            "health_record_request": "Help the user access health records; choose appropriate record_type parameter.",
            "symptom_triage": "Start symptom checker; never diagnose.",
            "find_medicine": "Guide to pharmacy search; never prescribe.",
            "prescription_inquiry": "Fetch prescription details; explain simply.",
            "medicine_scan": "Start medicine scanner; add safety disclaimers.",
            "emergency_assistance": "Trigger SOS with number 108 and short calming instructions.",
            "report_issue": "Guide to report/feedback flow.",
            "general_inquiry": "Briefly introduce capabilities and show features.",
            "out_of_scope": "Offer to connect to human support."
        }

        context_block = f"""
INTENT: {intent}
STAGE: {stage}
EMOTIONAL_STATE: {emotional_state}
URGENCY_LEVEL: {urgency_level}
LANGUAGE: {language}
GUIDANCE: {intent_guidance.get(intent, "Provide general navigation help for the app.")}
"""

        return f"{self.base_system_prompt}\n{context_block}\nRemember: Output ONLY a single valid JSON object."

    def analyze_user_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        if not self.is_available:
            return None
        try:
            analysis_prompt = f"""
Analyze the user's message for the Sehat Sahara health app and return ONLY a JSON object with:

{{
  "primary_intent": "one of: appointment_booking, appointment_view, appointment_cancel, health_record_request, symptom_triage, find_medicine, prescription_inquiry, medicine_scan, emergency_assistance, report_issue, general_inquiry, out_of_scope",
  "language_detected": "pa | hi | en",
  "urgency_level": "low | medium | high | emergency",
  "confidence": 0.0 to 1.0,
  "context_entities": {{"record_type":"all|labs|prescriptions|imaging", "specialty":"e.g., general_physician|pediatrician", "...": "..."}}
}}

Rules:
- Use "emergency_assistance" for severe terms like chest pain, unconsciousness, severe bleeding, stroke signs, trouble breathing.
- NEVER provide medical advice.
- If message asks for diagnosis/treatment, prefer appointment_booking with reason parameter.
- Detect language by content: Punjabi (pa), Hindi (hi), English (en).
- Return ONLY valid JSON, no extra text.

User message: {user_message}
"""
            response = self.client.generate_response(prompt=analysis_prompt, max_tokens=220, temperature=0.2)
            if response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                    # Minimal validation
                    if "primary_intent" in analysis and "language_detected" in analysis and "urgency_level" in analysis:
                        return analysis
        except Exception as e:
            self.logger.error(f"Error in Sehat Sahara intent analysis: {e}")
        return None

    def _fallback_action_response(self, language: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        # Provide a default response based on language and intent analysis
        lang = (intent_result.get("language_detected") or language or "hi")
        intent = intent_result.get("primary_intent", "general_inquiry")
        urgency = intent_result.get("urgency_level", "low")

        # Simple mapping to ensure a valid JSON response
        mapping = {
            "appointment_booking": ("NAVIGATE_TO_APPOINTMENT_BOOKING", {}),
            "appointment_view": ("FETCH_APPOINTMENTS", {}),
            "appointment_cancel": ("INITIATE_APPOINTMENT_CANCELLATION", {}),
            "health_record_request": ("FETCH_HEALTH_RECORD", {"record_type": "all"}),
            "symptom_triage": ("START_SYMPTOM_CHECKER", {}),
            "find_medicine": ("NAVIGATE_TO_PHARMACY_SEARCH", {}),
            "prescription_inquiry": ("FETCH_PRESCRIPTION_DETAILS", {}),
            "medicine_scan": ("START_MEDICINE_SCANNER", {}),
            "emergency_assistance": ("TRIGGER_SOS", {"emergency_number": "108", "type": "medical_emergency"}),
            "report_issue": ("NAVIGATE_TO_REPORT_ISSUE", {}),
            "general_inquiry": ("SHOW_APP_FEATURES", {}),
            "out_of_scope": ("CONNECT_TO_SUPPORT_AGENT", {"reason": "out_of_scope"}),
        }
        action, params = mapping.get(intent, mapping["general_inquiry"])

        # Short localized copy
        localized = {
            "en": "I'll guide you in the app.",
            "hi": "Main app mein aapki madad karti hoon.",
            "pa": "Main app vich tuhadi madad karangi.",
        }
        response_text = localized.get(lang, localized["hi"])

        if urgency == "emergency":
            action, params = mapping["emergency_assistance"]

        return {"response": response_text, "action": action, "parameters": params}

# Global instances for easy import
sehat_sahara_client = SehatSaharaApiClient()  # new canonical instance
groq_scout = GroqScoutClient()

# Legacy/compatibility aliases
api_llama3 = sehat_sahara_client
ollama_llama3 = sehat_sahara_client

# Legacy functions updated to use the new client
def generate_response(prompt: str, system_prompt: str = "") -> Optional[str]:
    return sehat_sahara_client.client.generate_response(prompt, system_prompt)

def is_llama_available() -> bool:
    return sehat_sahara_client.is_available

def test_llama_connection() -> Dict[str, Any]:
    return sehat_sahara_client.client.test_connection()

def get_llama_health() -> Dict[str, Any]:
    return sehat_sahara_client.get_status()
