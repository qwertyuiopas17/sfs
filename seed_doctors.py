# seed_database.py
from chatbot import app, db, Doctor, User  # Import necessary components from your main app
import bcrypt

# --- IMPORTANT ---
# This script should be run ONLY ONCE to add initial data.
# Running it again will cause an error if the doctors already exist.

def seed_data():
    """Adds initial sample doctors to the database."""
    with app.app_context():
        print("Connecting to the database to add doctors...")

        # List of sample doctors to add
        doctors_to_add = [
            {
                "doctor_id": "DOC001",
                "full_name": "Aarav Sharma",
                "specialization": "Cardiologist",
                "email": "aarav.sharma@clinic.com",
                "password": "password123",
                "profile_image": "https://images.unsplash.com/photo-1612349317150-e413f6a5b16e?q=80&w=2070&auto=format&fit=crop"
            },
            {
                "doctor_id": "DOC002",
                "full_name": "Priya Singh",
                "specialization": "Dermatologist",
                "email": "priya.singh@clinic.com",
                "password": "password123",
                "profile_image": "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=2070&auto=format&fit=crop"
            },
            {
                "doctor_id": "DOC003",
                "full_name": "Rohan Mehta",
                "specialization": "Pediatrician",
                "email": "rohan.mehta@clinic.com",
                "password": "password123",
                "profile_image": "https://images.unsplash.com/photo-1537368910025-700350796527?q=80&w=2070&auto=format&fit=crop"
            }
        ]

        for doc_data in doctors_to_add:
            # Check if a doctor with this ID already exists
            existing_doctor = Doctor.query.filter_by(doctor_id=doc_data["doctor_id"]).first()
            if not existing_doctor:
                hashed_password = bcrypt.hashpw(doc_data["password"].encode('utf-8'), bcrypt.gensalt())
                
                new_doctor = Doctor(
                    doctor_id=doc_data["doctor_id"],
                    full_name=doc_data["full_name"],
                    specialty=doc_data["specialization"],
                    email=doc_data["email"],
                    password_hash=hashed_password.decode('utf-8'),
                    profile_image_url=doc_data["profile_image"],
                    is_active=True
                )
                db.session.add(new_doctor)
                print(f"Adding Dr. {doc_data['full_name']}...")
            else:
                print(f"Dr. {doc_data['full_name']} already exists. Skipping.")

        # Commit all the new doctors to the database
        db.session.commit()
        print("\n✅ Successfully added doctors to the database!")

if __name__ == "__main__":
    seed_data()