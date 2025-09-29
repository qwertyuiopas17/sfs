import os
import argparse
from flask import Flask
from datetime import timedelta
from enhanced_database_models import db, User

def make_app():
    app = Flask(__name__)
    database_url = os.environ.get('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        db_uri = database_url.replace('postgres://', 'postgresql://', 1)
    else:
        # fallback to same sqlite path as chatbot.py uses
        basedir = os.path.abspath(os.path.dirname(__file__))
        instance_path = os.path.join(basedir, 'instance')
        os.makedirs(instance_path, exist_ok=True)
        db_uri = f'sqlite:///{os.path.join(instance_path, "enhanced_chatbot.db")}'
    app.config.update(
        SQLALCHEMY_DATABASE_URI=db_uri,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
        SECRET_KEY=os.environ.get('SECRET_KEY', os.urandom(24)),
    )
    db.init_app(app)
    return app

def main():
    parser = argparse.ArgumentParser(description="Create or update an admin user for Sehat Sahara.")
    parser.add_argument("--email", required=True, help="Admin email (unique)")
    parser.add_argument("--password", required=True, help="Admin password")
    parser.add_argument("--full-name", required=True, help="Admin full name")
    parser.add_argument("--phone", default="", help="Admin phone number")
    parser.add_argument("--patient-id", default="", help="Optional custom patient ID, auto-generated if omitted")
    args = parser.parse_args()

    app = make_app()
    with app.app_context():
        # Ensure tables exist
        db.create_all()

        # Check if user exists by email
        user = User.query.filter_by(email=args.email.lower().strip()).first()
        if user:
            user.full_name = args.full_name.strip()
            user.phone_number = args.phone.strip()
            user.role = 'admin'
            user.set_password(args.password)
            db.session.commit()
            print(f"✅ Updated existing user as admin: {user.email}")
            return

        # Auto-generate patient_id if missing
        patient_id = args.patient_id.strip()
        if not patient_id:
            last = User.query.order_by(User.id.desc()).first()
            seq = last.id + 1 if last else 1
            patient_id = f"PAT{str(seq).zfill(6)}"

        new_admin = User(
            patient_id=patient_id,
            email=args.email.lower().strip(),
            full_name=args.full_name.strip(),
            phone_number=args.phone.strip(),
            role='admin'
        )
        new_admin.set_password(args.password)
        db.session.add(new_admin)
        db.session.commit()
        print(f"✅ Created admin user: {new_admin.email} (patient_id={new_admin.patient_id})")

if __name__ == "__main__":
    main()
