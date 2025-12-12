import yaml
import pandas as pd
import os

def create_density_db(yaml_path, output_path):
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        names = data.get('names', [])

    print(f"Found {len(names)} classes.")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Food': names,
        'Density': [1.0] * len(names) # Default density
    })
    
    # Save to Excel
    df.to_excel(output_path, sheet_name='Density DB', index=False, header=False) # No header as per original code expectation (col 0, col 1)? 
    # Original code: pd.read_excel(..., usecols=[0, 1])
    # Original code doesn't specify header=None, so it likely expects a header or handles it.
    # Ah, "Food types are expected to be in column 1, food densities in column 2."
    # Let's verify original density_db loader:
    # pd.read_excel(db_path, sheet_name='Density DB', usecols=[0, 1])
    # Usually this reads the first row as header.
    # If I check `volume_estimator.py`:
    # match = process.extractOne(food, self.density_database.values[:,0])
    # This implies it uses the values regardless of header.
    # I will write "Food" and "Density" as header for clarity.

    # Wait, if I write header, I should check if the original code skips it.
    # Original code: pd.read_excel(...). 
    # If there is a header, it consumes the first row.
    # I'll create it with a header.
    
    df.to_excel(output_path, sheet_name='Density DB', index=False)
    print(f"Created {output_path}")

if __name__ == '__main__':
    create_density_db('models/FoodSeg.yaml', 'density_db.xlsx')
