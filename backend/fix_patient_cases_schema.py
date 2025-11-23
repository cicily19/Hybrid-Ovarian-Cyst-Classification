"""
Migration script to fix patient_cases table schema.
Removes patient_name, age, date_of_scan, symptoms columns (moved to patients table).
Converts patient_id from VARCHAR to INTEGER.
Removes old columns: verification_passed, p_simple, p_complex, timestamp.
"""
import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

print("Checking current schema...")
cursor.execute("PRAGMA table_info(patient_cases)")
columns = cursor.fetchall()
print(f"Current columns: {[col[1] for col in columns]}")

# Check if patient_name exists and is NOT NULL
patient_name_exists = any(col[1] == 'patient_name' for col in columns)
patient_name_not_null = any(col[1] == 'patient_name' and col[3] == 1 for col in columns)

if patient_name_exists:
    print("\n⚠️  Found patient_name column in patient_cases table.")
    print("This column should be removed as patient info is now in the patients table.")
    
    # First, make patient_name nullable if it's NOT NULL
    if patient_name_not_null:
        print("\nStep 1: Making patient_name nullable...")
        try:
            # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
            # But first, let's check if there are any rows
            cursor.execute("SELECT COUNT(*) FROM patient_cases")
            count = cursor.fetchone()[0]
            print(f"Found {count} rows in patient_cases")
            
            if count > 0:
                print("\n⚠️  WARNING: There are existing rows. We'll need to preserve data.")
                print("Creating backup table...")
                
                # Create backup
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patient_cases_backup AS 
                    SELECT * FROM patient_cases
                """)
                print("✓ Backup created")
            
            # Get all column names except patient_name, age, date_of_scan, symptoms
            cursor.execute("PRAGMA table_info(patient_cases)")
            all_cols = cursor.fetchall()
            cols_to_keep = [
                col[1] for col in all_cols 
                if col[1] not in ['patient_name', 'age', 'date_of_scan', 'symptoms', 
                                   'verification_passed', 'p_simple', 'p_complex', 'timestamp']
            ]
            
            # Also need to handle patient_id conversion from VARCHAR to INTEGER
            print("\nStep 2: Recreating patient_cases table with correct schema...")
            
            # Drop old table
            cursor.execute("DROP TABLE IF EXISTS patient_cases_old")
            cursor.execute("ALTER TABLE patient_cases RENAME TO patient_cases_old")
            
            # Create new table with correct schema
            cursor.execute("""
                CREATE TABLE patient_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    patient_id INTEGER NOT NULL,
                    image_path VARCHAR NOT NULL,
                    shap_path VARCHAR,
                    verification_score REAL,
                    prob_simple REAL,
                    prob_complex REAL,
                    prediction_label INTEGER,
                    predicted_class VARCHAR,
                    created_at DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """)
            
            # Migrate data if there are existing rows
            if count > 0:
                print("Step 3: Migrating data...")
                # Map old patient_id (VARCHAR) to new patient_id (INTEGER)
                # We need to find the patient_id in the patients table
                cursor.execute("""
                    INSERT INTO patient_cases 
                    (id, user_id, patient_id, image_path, shap_path, verification_score, 
                     prob_simple, prob_complex, prediction_label, predicted_class, created_at)
                    SELECT 
                        old.id,
                        old.user_id,
                        p.id as patient_id,
                        old.image_path,
                        old.shap_path,
                        old.verification_score,
                        CASE 
                            WHEN old.prob_simple IS NOT NULL THEN CAST(old.prob_simple AS REAL)
                            WHEN old.p_simple IS NOT NULL THEN CAST(old.p_simple AS REAL)
                            ELSE old.prob_simple
                        END as prob_simple,
                        CASE 
                            WHEN old.prob_complex IS NOT NULL THEN CAST(old.prob_complex AS REAL)
                            WHEN old.p_complex IS NOT NULL THEN CAST(old.p_complex AS REAL)
                            ELSE old.prob_complex
                        END as prob_complex,
                        old.prediction_label,
                        old.predicted_class,
                        CASE 
                            WHEN old.created_at IS NOT NULL THEN old.created_at
                            WHEN old.timestamp IS NOT NULL THEN old.timestamp
                            ELSE datetime('now')
                        END as created_at
                    FROM patient_cases_old old
                    LEFT JOIN patients p ON p.patient_id = old.patient_id
                    WHERE p.id IS NOT NULL
                """)
                
                migrated_count = cursor.rowcount
                print(f"✓ Migrated {migrated_count} rows")
                
                if migrated_count < count:
                    print(f"⚠️  WARNING: Only {migrated_count} of {count} rows were migrated.")
                    print("Some rows may not have matching patients.")
            
            # Drop old table
            cursor.execute("DROP TABLE patient_cases_old")
            print("✓ Old table removed")
            
            conn.commit()
            print("\n✅ Migration completed successfully!")
            
        except Exception as e:
            conn.rollback()
            print(f"\n❌ Error during migration: {e}")
            print("Rolling back changes...")
            # Restore from backup if it exists
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient_cases_backup'")
                if cursor.fetchone():
                    cursor.execute("DROP TABLE IF EXISTS patient_cases")
                    cursor.execute("ALTER TABLE patient_cases_backup RENAME TO patient_cases")
                    conn.commit()
                    print("✓ Restored from backup")
            except:
                pass
            raise
    else:
        print("patient_name is already nullable. No migration needed.")
else:
    print("\n✅ patient_name column not found. Schema appears to be correct.")

# Verify final schema
print("\nVerifying final schema...")
cursor.execute("PRAGMA table_info(patient_cases)")
final_columns = cursor.fetchall()
print(f"Final columns: {[col[1] for col in final_columns]}")

conn.close()
print("\n✅ Done!")


