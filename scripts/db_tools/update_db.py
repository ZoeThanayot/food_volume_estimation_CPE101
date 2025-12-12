import pandas as pd
import os

def update_density_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    # Read existing DB
    df = pd.read_excel(db_path)
    
    # Check current columns
    print(f"Current columns: {df.columns.tolist()}")
    
    # New columns to add
    new_cols = ['kcal', 'protein', 'carbs', 'fat', 'sugar']
    
    for col in new_cols:
        if col not in df.columns:
            print(f"Adding column: {col}")
            df[col] = 0.0
            
    # Save back
    df.to_excel(db_path, index=False)
    print(f"Updated {db_path} with new columns.")
    print(f"New columns: {df.columns.tolist()}")

if __name__ == '__main__':
    update_density_db('density_db.xlsx')
