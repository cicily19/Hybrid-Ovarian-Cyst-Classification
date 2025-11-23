import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(patient_cases)")
case_cols = [(row[1], row[2]) for row in cursor.fetchall()]
print("Patient_cases columns (name, type):", case_cols)

# Check if patient_id is TEXT or INTEGER
for col_name, col_type in case_cols:
    if col_name == 'patient_id':
        print(f"\npatient_id column type: {col_type}")
        if col_type.upper() == 'TEXT':
            print("WARNING: patient_id is TEXT, should be INTEGER for foreign key")
        break

conn.close()
