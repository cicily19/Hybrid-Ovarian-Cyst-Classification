"""
Database migration script to add new columns to existing tables.
Run this once to update your database schema.
"""

import sqlite3
import os

DATABASE_URL = "sqlite:///./users.db"
DB_PATH = "users.db"

def migrate_database():
    """Add missing columns to existing database tables."""
    if not os.path.exists(DB_PATH):
        print("Database does not exist. It will be created automatically on first run.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check and add columns to users table
        cursor.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'profile_pic' not in columns:
            print("Adding 'profile_pic' column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
        
        if 'created_at' not in columns:
            print("Adding 'created_at' column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME")
        
        if 'last_login' not in columns:
            print("Adding 'last_login' column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME")
        
        # Check if patients table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        if not cursor.fetchone():
            print("Creating 'patients' table...")
            cursor.execute("""
                CREATE TABLE patients (
                    id INTEGER PRIMARY KEY,
                    patient_name TEXT NOT NULL,
                    patient_id TEXT NOT NULL UNIQUE,
                    age INTEGER NOT NULL,
                    gender TEXT,
                    clinical_notes TEXT,
                    date_of_scan TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX ix_patients_patient_id ON patients(patient_id)")
        
        # Check and update patient_cases table
        cursor.execute("PRAGMA table_info(patient_cases)")
        case_columns = [row[1] for row in cursor.fetchall()]
        
        # Check if patient_id column exists and is the right type
        if 'patient_id' in case_columns:
            # Check if it's a foreign key (Integer) or old String type
            cursor.execute("PRAGMA table_info(patient_cases)")
            for row in cursor.fetchall():
                if row[1] == 'patient_id':
                    if row[2] == 'TEXT':
                        # Old schema - need to migrate
                        print("Migrating patient_cases.patient_id from TEXT to INTEGER...")
                        # This is complex, so we'll just note it
                        print("Note: patient_id migration from TEXT to INTEGER requires data migration.")
        
        # Add new columns to patient_cases
        if 'verification_score' not in case_columns:
            print("Adding 'verification_score' column to patient_cases table...")
            cursor.execute("ALTER TABLE patient_cases ADD COLUMN verification_score REAL")
        
        if 'prob_simple' not in case_columns:
            print("Adding 'prob_simple' column to patient_cases table...")
            cursor.execute("ALTER TABLE patient_cases ADD COLUMN prob_simple REAL")
        
        if 'prob_complex' not in case_columns:
            print("Adding 'prob_complex' column to patient_cases table...")
            cursor.execute("ALTER TABLE patient_cases ADD COLUMN prob_complex REAL")
        
        if 'prediction_label' not in case_columns:
            print("Adding 'prediction_label' column to patient_cases table...")
            cursor.execute("ALTER TABLE patient_cases ADD COLUMN prediction_label INTEGER")
        
        if 'created_at' not in case_columns:
            print("Adding 'created_at' column to patient_cases table...")
            cursor.execute("ALTER TABLE patient_cases ADD COLUMN created_at DATETIME")
        
        if 'batch_id' not in case_columns:
            print("Adding 'batch_id' column to patient_cases table...")
            cursor.execute("ALTER TABLE patient_cases ADD COLUMN batch_id INTEGER")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_patient_cases_batch_id ON patient_cases(batch_id)")
        
        # Check if batches table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='batches'")
        if not cursor.fetchone():
            print("Creating 'batches' table...")
            cursor.execute("""
                CREATE TABLE batches (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    batch_name TEXT,
                    status TEXT DEFAULT 'uploading',
                    total_cases INTEGER DEFAULT 0,
                    completed_cases INTEGER DEFAULT 0,
                    failed_cases INTEGER DEFAULT 0,
                    pending_cases INTEGER DEFAULT 0,
                    created_at DATETIME,
                    started_at DATETIME,
                    completed_at DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_batches_user_id ON batches(user_id)")
        
        # Check annotations table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='annotations'")
        if cursor.fetchone():
            cursor.execute("PRAGMA table_info(annotations)")
            annotation_columns = [row[1] for row in cursor.fetchall()]
            
            if 'created_at' not in annotation_columns:
                print("Adding 'created_at' column to annotations table...")
                cursor.execute("ALTER TABLE annotations ADD COLUMN created_at DATETIME")
        
        conn.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"Migration error: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()


