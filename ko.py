"""
Sehat Sahara Health Assistant Response Generator
Generates action-oriented responses for health app navigation
Supports Punjabi, Hindi, and English for rural patients
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

class ProgressiveResponseGenerator:
    """
    Response generator for Sehat Sahara Health Assistant.
    Generates functional guidance responses with app actions.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Multilingual responses for each intent with corresponding actions
        self.intent_responses = {
            'appointment_booking': {
                'en': {
                    'responses': [
                        "I can help you book an appointment with a doctor. Let me guide you to the booking section.",
                        "I'll help you schedule a consultation. Which type of doctor would you like to see?",
                        "Let's book your appointment. I'll take you to the doctor selection page."
                    ],
                    'action': 'NAVIGATE_TO_APPOINTMENT_BOOKING',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main aapko doctor ke saath appointment book karne mein madad kar sakti hoon. Aapko kis doctor se milna hai?",
                        "Appointment book karne ke liye main aapki madad karungi. Kya aap bata sakte hain kis prakar ke doctor chahiye?",
                        "Chaliye appointment book karte hain. Main aapko doctor selection page par le chalti hoon."
                    ],
                    'action': 'NAVIGATE_TO_APPOINTMENT_BOOKING',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main tuhanu doctor naal appointment book karan vich madad kar sakdi haan. Tuhanu kis doctor nu milna hai?",
                        "Appointment book karan layi main tuhadi madad karangi. Ki tusi dass sakde ho kis tarah de doctor chahide?",
                        "Chalo appointment book karde haan. Main tuhanu doctor selection page te le chalti haan."
                    ],
                    'action': 'NAVIGATE_TO_APPOINTMENT_BOOKING',
                    'parameters': {}
                }
            },
            'appointment_view': {
                'en': {
                    'responses': [
                        "Let me show you your upcoming appointments.",
                        "I'll fetch your appointment details for you.",
                        "Here are your scheduled appointments."
                    ],
                    'action': 'FETCH_APPOINTMENTS',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main aapki upcoming appointments dikhati hoon.",
                        "Aapki appointment ki details main le kar aati hoon.",
                        "Ye hain aapki scheduled appointments."
                    ],
                    'action': 'FETCH_APPOINTMENTS',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main tuhadi upcoming appointments dikhandi haan.",
                        "Tuhadi appointment dian details main le ke aandi haan.",
                        "Eh hain tuhadian scheduled appointments."
                    ],
                    'action': 'FETCH_APPOINTMENTS',
                    'parameters': {}
                }
            },
            'appointment_cancel': {
                'en': {
                    'responses': [
                        "I'll help you cancel your appointment. Let me show you your bookings.",
                        "To cancel an appointment, I need to show you your current bookings first.",
                        "Let me guide you through the cancellation process."
                    ],
                    'action': 'INITIATE_APPOINTMENT_CANCELLATION',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main aapki appointment cancel karne mein madad karungi. Pehle aapki bookings dikhati hoon.",
                        "Appointment cancel karne ke liye pehle main aapki current bookings dikhaungi.",
                        "Main aapko cancellation process guide karungi."
                    ],
                    'action': 'INITIATE_APPOINTMENT_CANCELLATION',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main tuhadi appointment cancel karan vich madad karangi. Pehlan tuhadian bookings dikhandi haan.",
                        "Appointment cancel karan layi pehlan main tuhadian current bookings dikhaungi.",
                        "Main tuhanu cancellation process guide karangi."
                    ],
                    'action': 'INITIATE_APPOINTMENT_CANCELLATION',
                    'parameters': {}
                }
            },
            'health_record_request': {
                'en': {
                    'responses': [
                        "I'll fetch your health records for you.",
                        "Let me show you your medical reports and history.",
                        "Accessing your health records now."
                    ],
                    'action': 'FETCH_HEALTH_RECORD',
                    'parameters': {'record_type': 'all'}
                },
                'hi': {
                    'responses': [
                        "Main aapke health records le kar aati hoon.",
                        "Aapki medical reports aur history dikhati hoon.",
                        "Aapke health records access kar rahi hoon."
                    ],
                    'action': 'FETCH_HEALTH_RECORD',
                    'parameters': {'record_type': 'all'}
                },
                'pa': {
                    'responses': [
                        "Main tuhade health records le ke aandi haan.",
                        "Tuhadian medical reports te history dikhandi haan.",
                        "Tuhade health records access kar rahi haan."
                    ],
                    'action': 'FETCH_HEALTH_RECORD',
                    'parameters': {'record_type': 'all'}
                }
            },
            'symptom_triage': {
                'en': {
                    'responses': [
                        "I understand you're not feeling well. Let me help you with a symptom checker.",
                        "Let's assess your symptoms. I'll guide you through some questions.",
                        "I'll help you understand your symptoms better."
                    ],
                    'action': 'START_SYMPTOM_CHECKER',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main samajh gayi ki aapki tabiyat theek nahi hai. Symptom checker se madad karte hain.",
                        "Aapke symptoms check karte hain. Main kuch sawal puchungi.",
                        "Main aapke symptoms samjhane mein madad karungi."
                    ],
                    'action': 'START_SYMPTOM_CHECKER',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main samajh gayi ki tuhadi tabiyat theek nahi hai. Symptom checker naal madad karde haan.",
                        "Tuhade symptoms check karde haan. Main kujh sawal puchhangi.",
                        "Main tuhade symptoms samjhan vich madad karangi."
                    ],
                    'action': 'START_SYMPTOM_CHECKER',
                    'parameters': {}
                }
            },
            'find_medicine': {
                'en': {
                    'responses': [
                        "I'll help you find nearby pharmacies where you can get your medicine.",
                        "Let me show you medicine shops in your area.",
                        "I'll guide you to find the medicine you need."
                    ],
                    'action': 'NAVIGATE_TO_PHARMACY_SEARCH',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main aapko paas ki pharmacy dhundne mein madad karungi jahan aapko medicine mil sakti hai.",
                        "Aapke area mein medicine shops dikhati hoon.",
                        "Jo medicine chahiye usse dhundne mein madad karungi."
                    ],
                    'action': 'NAVIGATE_TO_PHARMACY_SEARCH',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main tuhanu nazdeeki pharmacy labhan vich madad karangi jithe tuhanu medicine mil sakdi hai.",
                        "Tuhade area vich medicine shops dikhandi haan.",
                        "Jo medicine chahidi usse labhan vich madad karangi."
                    ],
                    'action': 'NAVIGATE_TO_PHARMACY_SEARCH',
                    'parameters': {}
                }
            },
            'prescription_inquiry': {
                'en': {
                    'responses': [
                        "I'll show you the details of your prescription and how to take your medicines.",
                        "Let me fetch your prescription information.",
                        "I'll help you understand your medicine instructions."
                    ],
                    'action': 'FETCH_PRESCRIPTION_DETAILS',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main aapke prescription ki details aur medicine kaise leni hai ye dikhati hoon.",
                        "Aapki prescription ki jankari le kar aati hoon.",
                        "Medicine ki instructions samjhane mein madad karungi."
                    ],
                    'action': 'FETCH_PRESCRIPTION_DETAILS',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main tuhade prescription dian details te medicine kive leni hai eh dikhandi haan.",
                        "Tuhadi prescription di jankari le ke aandi haan.",
                        "Medicine dian instructions samjhan vich madad karangi."
                    ],
                    'action': 'FETCH_PRESCRIPTION_DETAILS',
                    'parameters': {}
                }
            },
            'medicine_scan': {
                'en': {
                    'responses': [
                        "I'll help you scan and identify your medicine. Please use the camera feature.",
                        "Let me guide you to the medicine scanner.",
                        "Use the scanner to identify your medicine."
                    ],
                    'action': 'START_MEDICINE_SCANNER',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main aapki medicine scan aur identify karne mein madad karungi. Camera feature use kariye.",
                        "Medicine scanner tak le chalti hoon.",
                        "Medicine identify karne ke liye scanner use kariye."
                    ],
                    'action': 'START_MEDICINE_SCANNER',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main tuhadi medicine scan te identify karan vich madad karangi. Camera feature use karo.",
                        "Medicine scanner tak le chalti haan.",
                        "Medicine identify karan layi scanner use karo."
                    ],
                    'action': 'START_MEDICINE_SCANNER',
                    'parameters': {}
                }
            },
            'emergency_assistance': {
                'en': {
                    'responses': [
                        "This is an emergency situation. I'm connecting you to emergency services immediately. For ambulance, call 108.",
                        "Emergency detected! Please call 108 for ambulance or go to the nearest hospital immediately.",
                        "I'm triggering emergency assistance. Ambulance number: 108. Stay calm, help is coming."
                    ],
                    'action': 'TRIGGER_SOS',
                    'parameters': {'emergency_number': '108', 'type': 'medical_emergency'}
                },
                'hi': {
                    'responses': [
                        "Ye emergency situation hai. Main aapko turant emergency services se connect kar rahi hoon. Ambulance ke liye 108 call kariye.",
                        "Emergency detect hui hai! Ambulance ke liye 108 call kariye ya nazdeeki hospital jaldi jaiye.",
                        "Main emergency assistance trigger kar rahi hoon. Ambulance number: 108. Ghabraiye mat, madad aa rahi hai."
                    ],
                    'action': 'TRIGGER_SOS',
                    'parameters': {'emergency_number': '108', 'type': 'medical_emergency'}
                },
                'pa': {
                    'responses': [
                        "Eh emergency situation hai. Main tuhanu turant emergency services naal connect kar rahi haan. Ambulance layi 108 call karo.",
                        "Emergency detect hoyi hai! Ambulance layi 108 call karo ya nazdeeki hospital jaldi jao.",
                        "Main emergency assistance trigger kar rahi haan. Ambulance number: 108. Ghabrao nahi, madad aa rahi hai."
                    ],
                    'action': 'TRIGGER_SOS',
                    'parameters': {'emergency_number': '108', 'type': 'medical_emergency'}
                }
            },
            'report_issue': {
                'en': {
                    'responses': [
                        "I'm sorry to hear about your experience. Let me help you report this issue.",
                        "I'll guide you to the feedback section where you can report this problem.",
                        "Your feedback is important. Let me take you to the complaint section."
                    ],
                    'action': 'NAVIGATE_TO_REPORT_ISSUE',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Aapke experience ke baare mein sunkar dukh hua. Main is issue report karne mein madad karungi.",
                        "Feedback section mein le chalti hoon jahan aap ye problem report kar sakte hain.",
                        "Aapka feedback important hai. Complaint section mein le chalti hoon."
                    ],
                    'action': 'NAVIGATE_TO_REPORT_ISSUE',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Tuhade experience bare sunke dukh hoya. Main is issue report karan vich madad karangi.",
                        "Feedback section vich le chalti haan jithe tusi eh problem report kar sakde ho.",
                        "Tuhada feedback important hai. Complaint section vich le chalti haan."
                    ],
                    'action': 'NAVIGATE_TO_REPORT_ISSUE',
                    'parameters': {}
                }
            },
            'general_inquiry': {
                'en': {
                    'responses': [
                        "I'm here to help you navigate the Sehat Sahara app. I can help you book appointments, find medicines, check your health records, and more.",
                        "Welcome to Sehat Sahara! I can assist you with appointments, health records, finding pharmacies, and emergency help.",
                        "I'm your health assistant. I can help you with doctor appointments, medicine information, symptom checking, and app navigation."
                    ],
                    'action': 'SHOW_APP_FEATURES',
                    'parameters': {}
                },
                'hi': {
                    'responses': [
                        "Main Sehat Sahara app navigate karne mein aapki madad ke liye hoon. Appointment book karna, medicine dhundna, health records check karna - sab mein madad kar sakti hoon.",
                        "Sehat Sahara mein aapka swagat hai! Main appointments, health records, pharmacy dhundne, aur emergency help mein madad kar sakti hoon.",
                        "Main aapki health assistant hoon. Doctor appointments, medicine ki jankari, symptom checking, aur app navigation mein madad kar sakti hoon."
                    ],
                    'action': 'SHOW_APP_FEATURES',
                    'parameters': {}
                },
                'pa': {
                    'responses': [
                        "Main Sehat Sahara app navigate karan vich tuhadi madad layi haan. Appointment book karna, medicine labhna, health records check karna - sab vich madad kar sakdi haan.",
                        "Sehat Sahara vich tuhada swagat hai! Main appointments, health records, pharmacy labhne, te emergency help vich madad kar sakdi haan.",
                        "Main tuhadi health assistant haan. Doctor appointments, medicine di jankari, symptom checking, te app navigation vich madad kar sakdi haan."
                    ],
                    'action': 'SHOW_APP_FEATURES',
                    'parameters': {}
                }
            },
            'out_of_scope': {
                'en': {
                    'responses': [
                        "I'm designed to help with health-related queries and app navigation. For other questions, would you like to talk to a human Sehat Saathi?",
                        "I can only assist with health and medical app features. Would you like me to connect you with a support agent for other queries?",
                        "I focus on health assistance. For non-health related questions, I can connect you with our support team."
                    ],
                    'action': 'CONNECT_TO_SUPPORT_AGENT',
                    'parameters': {'reason': 'out_of_scope'}
                },
                'hi': {
                    'responses': [
                        "Main health-related queries aur app navigation mein madad ke liye banayi gayi hoon. Dusre sawalon ke liye kya aap human Sehat Saathi se baat karna chahenge?",
                        "Main sirf health aur medical app features mein madad kar sakti hoon. Dusre queries ke liye support agent se connect karun?",
                        "Main health assistance par focus karti hoon. Non-health related questions ke liye main aapko support team se connect kar sakti hoon."
                    ],
                    'action': 'CONNECT_TO_SUPPORT_AGENT',
                    'parameters': {'reason': 'out_of_scope'}
                },
                'pa': {
                    'responses': [
                        "Main health-related queries te app navigation vich madad layi banayi gayi haan. Dusre sawaalan layi ki tusi human Sehat Saathi naal gall karna chahoge?",
                        "Main sirf health te medical app features vich madad kar sakdi haan. Dusre queries layi support agent naal connect karan?",
                        "Main health assistance te focus kardi haan. Non-health related questions layi main tuhanu support team naal connect kar sakdi haan."
                    ],
                    'action': 'CONNECT_TO_SUPPORT_AGENT',
                    'parameters': {'reason': 'out_of_scope'}
                }
            }
        }

        # Medical advice safety responses
        self.medical_advice_responses = {
            'en': {
                'responses': [
                    "I cannot give medical advice, but I can help you connect with a qualified doctor. Would you like to book a consultation?",
                    "For medical advice, please consult with a qualified healthcare professional. I can help you book an appointment.",
                    "I'm not qualified to provide medical advice. Let me help you connect with a doctor who can properly assist you."
                ],
                'action': 'NAVIGATE_TO_APPOINTMENT_BOOKING',
                'parameters': {'reason': 'medical_advice_needed'}
            },
            'hi': {
                'responses': [
                    "Main medical advice nahi de sakti, lekin qualified doctor se connect karne mein madad kar sakti hoon. Kya aap consultation book karna chahenge?",
                    "Medical advice ke liye qualified healthcare professional se consult kariye. Main appointment book karne mein madad kar sakti hoon.",
                    "Main medical advice dene ke liye qualified nahi hoon. Doctor se connect karne mein madad karti hoon jo properly assist kar sake."
                ],
                'action': 'NAVIGATE_TO_APPOINTMENT_BOOKING',
                'parameters': {'reason': 'medical_advice_needed'}
            },
            'pa': {
                'responses': [
                    "Main medical advice nahi de sakdi, par qualified doctor naal connect karan vich madad kar sakdi haan. Ki tusi consultation book karna chahoge?",
                    "Medical advice layi qualified healthcare professional naal consult karo. Main appointment book karan vich madad kar sakdi haan.",
                    "Main medical advice den layi qualified nahi haan. Doctor naal connect karan vich madad kardi haan jo properly assist kar sake."
                ],
                'action': 'NAVIGATE_TO_APPOINTMENT_BOOKING',
                'parameters': {'reason': 'medical_advice_needed'}
            }
        }

        # Confusion/unclear request responses
        self.confusion_responses = {
            'en': {
                'responses': [
                    "I'm sorry, I didn't understand. Would you like to talk to a human 'Sehat Saathi' for help?",
                    "I couldn't quite understand your request. Let me connect you with a support agent who can better assist you.",
                    "I'm not sure how to help with that. Would you like me to connect you with our support team?"
                ],
                'action': 'CONNECT_TO_SUPPORT_AGENT',
                'parameters': {'reason': 'unclear_request'}
            },
            'hi': {
                'responses': [
                    "Maaf kariye, main samajh nahi payi. Kya aap human 'Sehat Saathi' se madad ke liye baat karna chahenge?",
                    "Aapki request samajh nahi aayi. Support agent se connect karti hoon jo better assist kar sake.",
                    "Main sure nahi hoon ki isme kaise madad karun. Support team se connect kar dun?"
                ],
                'action': 'CONNECT_TO_SUPPORT_AGENT',
                'parameters': {'reason': 'unclear_request'}
            },
            'pa': {
                'responses': [
                    "Maaf karo, main samajh nahi payi. Ki tusi human 'Sehat Saathi' naal madad layi gall karna chahoge?",
                    "Tuhadi request samajh nahi aayi. Support agent naal connect kardi haan jo better assist kar sake.",
                    "Main sure nahi haan ki isme kive madad karan. Support team naal connect kar dun?"
                ],
                'action': 'CONNECT_TO_SUPPORT_AGENT',
                'parameters': {'reason': 'unclear_request'}
            }
        }

    def generate_response(self, 
                         user_message: str,
                         nlu_result: Dict[str, Any],
                         user_context: Dict[str, Any] = None,
                         conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate action-oriented response for Sehat Sahara Health Assistant.
        Returns a dictionary with 'response' text and 'action' for the mobile app.
        """
        
        try:
            intent = nlu_result.get('primary_intent', 'general_inquiry')
            language = nlu_result.get('language_detected', 'en')
            urgency = nlu_result.get('urgency_level', 'low')
            context_entities = nlu_result.get('context_entities', {})
            
            # Check if this is a medical advice request
            if self._is_medical_advice_request(user_message):
                return self._get_medical_advice_response(language)
            
            # Check if request is unclear
            if nlu_result.get('confidence', 0) < 0.3:
                return self._get_confusion_response(language)
            
            # Get appropriate response based on intent and language
            response_data = self._get_intent_response(intent, language, urgency, context_entities)
            
            # Add user context to parameters if available
            if user_context:
                response_data['parameters'].update({
                    'user_id': user_context.get('user_id'),
                    'session_id': user_context.get('session_id')
                })
            
            # Add conversation metadata
            response_data.update({
                'intent': intent,
                'language': language,
                'urgency_level': urgency,
                'timestamp': datetime.now().isoformat(),
                'confidence': nlu_result.get('confidence', 0.5)
            })
            
            self.logger.info(f"Generated response for intent: {intent}, language: {language}, action: {response_data.get('action')}")
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(language)

    def _is_medical_advice_request(self, message: str) -> bool:
        """Check if user is asking for medical advice."""
        medical_advice_keywords = [
            'what medicine should i take', 'which tablet is good', 'how to cure',
            'what treatment', 'diagnose', 'kya dawai lun', 'kya ilaj hai',
            'ki dawai leni chahidi', 'ki ilaj hai'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in medical_advice_keywords)

    def _get_medical_advice_response(self, language: str) -> Dict[str, Any]:
        """Get medical advice safety response."""
        lang_responses = self.medical_advice_responses.get(language, self.medical_advice_responses['en'])
        response_text = random.choice(lang_responses['responses'])
        
        return {
            'response': response_text,
            'action': lang_responses['action'],
            'parameters': lang_responses['parameters'],
            'safety_triggered': True
        }

    def _get_confusion_response(self, language: str) -> Dict[str, Any]:
        """Get response for unclear requests."""
        lang_responses = self.confusion_responses.get(language, self.confusion_responses['en'])
        response_text = random.choice(lang_responses['responses'])
        
        return {
            'response': response_text,
            'action': lang_responses['action'],
            'parameters': lang_responses['parameters'],
            'confusion_handled': True
        }

    def _get_intent_response(self, intent: str, language: str, urgency: str, context_entities: Dict) -> Dict[str, Any]:
        """Get response for specific intent."""
        intent_data = self.intent_responses.get(intent, self.intent_responses['general_inquiry'])
        lang_data = intent_data.get(language, intent_data['en'])
        
        response_text = random.choice(lang_data['responses'])
        action = lang_data['action']
        parameters = lang_data['parameters'].copy()
        
        # Add context-specific parameters
        if context_entities:
            parameters.update(context_entities)
        
        # Modify parameters based on urgency
        if urgency == 'emergency':
            parameters['priority'] = 'high'
            parameters['urgent'] = True
        
        return {
            'response': response_text,
            'action': action,
            'parameters': parameters
        }

    def _get_fallback_response(self, language: str = 'en') -> Dict[str, Any]:
        """Get fallback response for errors."""
        fallback_responses = {
            'en': "I'm having trouble right now. Let me connect you with a support agent.",
            'hi': "Mujhe abhi problem aa rahi hai. Support agent se connect kar deti hoon.",
            'pa': "Minu abhi problem aa rahi hai. Support agent naal connect kar dendi haan."
        }
        
        return {
            'response': fallback_responses.get(language, fallback_responses['en']),
            'action': 'CONNECT_TO_SUPPORT_AGENT',
            'parameters': {'reason': 'system_error'},
            'fallback_triggered': True
        }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ['en', 'hi', 'pa']

    def get_supported_actions(self) -> List[str]:
        """Get list of all supported actions."""
        actions = set()
        for intent_data in self.intent_responses.values():
            for lang_data in intent_data.values():
                if isinstance(lang_data, dict) and 'action' in lang_data:
                    actions.add(lang_data['action'])
        
        # Add additional actions
        actions.update([
            'CONNECT_TO_SUPPORT_AGENT',
            'SHOW_APP_FEATURES'
        ])
        
        return list(actions)

    def validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """Validate that response has required structure."""
        required_fields = ['response', 'action']
        return all(field in response for field in required_fields)

    def get_response_stats(self) -> Dict[str, Any]:
        """Get statistics about response generation."""
        return {
            'total_intents': len(self.intent_responses),
            'supported_languages': len(self.get_supported_languages()),
            'supported_actions': len(self.get_supported_actions()),
            'safety_responses_configured': len(self.medical_advice_responses),
            'confusion_responses_configured': len(self.confusion_responses)
        }