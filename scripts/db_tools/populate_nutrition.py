import pandas as pd
import os


# User provided data
# Format: [kcal, protein, carbs, fat, sugar]
nutrition_map = {
    'almond': [579, 21, 22, 50, 4],
    'apple': [52, 0.3, 14, 0.2, 10],
    'apricot': [48, 1.4, 11, 0.4, 9],
    'asparagus': [20, 2.2, 3.9, 0.1, 1.9],
    'avocado': [160, 2, 9, 15, 0.7],
    'bamboo shoots': [27, 2.6, 5, 0.3, 3],
    'banana': [89, 1.1, 23, 0.3, 12],
    'bean sprouts': [30, 3, 6, 0.2, 4],
    'biscuit': [353, 7, 75, 16, 18],
    'blueberry': [57, 0.7, 14, 0.3, 10],
    'bread': [265, 9, 49, 3.2, 5],
    'broccoli': [34, 2.8, 7, 0.4, 1.7],
    'cabbage': [25, 1.3, 6, 0.1, 3],
    'cake': [371, 5, 53, 15, 30],
    'candy': [394, 0, 98, 0.2, 60],
    'carrot': [41, 0.9, 10, 0.2, 4.7],
    'cashew': [553, 18, 30, 44, 6],
    'cauliflower': [25, 1.9, 5, 0.3, 1.9],
    'celery stick': [16, 0.7, 3, 0.2, 1.3],
    'cheese butter': [371, 25, 1.3, 33, 0.5], 
    'cherry': [50, 1, 12, 0.3, 8],
    'chicken duck': [239, 27, 0, 14, 0], 
    'chocolate': [546, 5, 61, 31, 48],
    'cilantro mint': [23, 2, 4, 0.5, 0],
    'coffee': [0, 0, 0, 0, 0], 
    'corn': [86, 3.2, 19, 1.2, 3],
    'crab': [97, 19, 0, 1.5, 0],
    'cucumber': [15, 0.7, 3.6, 0.1, 1.7],
    'date': [282, 2.5, 75, 0.4, 66],
    'dried cranberries': [308, 0, 82, 1.4, 65],
    'egg': [155, 13, 1.1, 11, 1.1],
    'eggplant': [25, 1, 6, 0.2, 3.5],
    'egg tart': [290, 5, 29, 17, 15],
    'enoki mushroom': [37, 2.7, 7.8, 0.3, 0],
    'fig': [74, 0.8, 19, 0.3, 16],
    'fish': [205, 22, 0, 12, 0],
    'french beans': [31, 1.8, 7, 0.2, 3],
    'french fries': [312, 3.4, 41, 15, 0.3],
    'fried meat': [280, 20, 10, 18, 0], 
    'garlic': [149, 6.4, 33, 0.5, 1],
    'ginger': [80, 1.8, 18, 0.8, 1.7],
    'grape': [69, 0.7, 18, 0.2, 15],
    'green beans': [31, 1.8, 7, 0.2, 3],
    'hamburg': [250, 26, 0, 15, 0],
    'hanamaki baozi': [250, 7, 50, 3, 5], 
    'ice cream': [207, 3.5, 24, 11, 21],
    'juice': [45, 0.5, 10, 0.1, 9],
    'kelp': [43, 1.7, 9.6, 0.6, 0.6],
    'king oyster mushroom': [35, 3, 5, 0.4, 1],
    'kiwi': [61, 1.1, 15, 0.5, 9],
    'lamb': [294, 25, 0, 21, 0],
    'lemon': [29, 1.1, 9, 0.3, 2.5],
    'lettuce': [15, 1.4, 2.9, 0.2, 0.8],
    'mango': [60, 0.8, 15, 0.4, 14],
    'melon': [34, 0.8, 8, 0.2, 8],
    'milk': [42, 3.4, 5, 1, 5],
    'milkshake': [112, 3, 18, 3, 18],
    'noodles': [138, 4.5, 25, 2.1, 0.5],
    'okra': [33, 1.9, 7, 0.2, 1.5],
    'olives': [115, 0.8, 6, 11, 0],
    'onion': [40, 1.1, 9, 0.1, 4.2],
    'orange': [47, 0.9, 12, 0.1, 9],
    'other ingredients': [0, 0, 0, 0, 0],
    'oyster mushroom': [33, 3.3, 6, 0.4, 1],
    'pasta': [131, 5, 25, 1.1, 0.6],
    'peach': [39, 0.9, 10, 0.3, 8],
    'peanut': [567, 26, 16, 49, 4],
    'pear': [57, 0.4, 15, 0.1, 10],
    'pepper': [20, 0.9, 4.6, 0.2, 2.4],
    'pie': [237, 3, 34, 11, 15],
    'pineapple': [50, 0.5, 13, 0.1, 10],
    'pizza': [266, 11, 33, 10, 3.6],
    'popcorn': [375, 11, 74, 4.3, 1],
    'pork': [242, 27, 0, 14, 0],
    'potato': [77, 2, 17, 0.1, 0.8],
    'pudding': [120, 3, 18, 4, 16],
    'pumpkin': [26, 1, 6.5, 0.1, 2.8],
    'rape': [22, 2.2, 3.8, 0.3, 0.5], 
    'raspberry': [52, 1.2, 12, 0.7, 4.4],
    'red beans': [127, 7.5, 23, 0.5, 0.3],
    'rice': [130, 2.7, 28, 0.3, 0.1],
    'salad': [17, 1.2, 3.3, 0.2, 2],
    'sauce': [80, 1, 15, 0.5, 10],
    'sausage': [300, 12, 2, 27, 1],
    'seaweed': [45, 1.7, 9, 0.6, 0.6],
    'shellfish': [100, 20, 2, 1.5, 0],
    'shiitake': [34, 2.2, 6.8, 0.5, 2],
    'shrimp': [99, 24, 0.2, 0.3, 0],
    'snow peas': [42, 2.8, 7.6, 0.2, 4],
    'soup': [35, 2, 4, 1.5, 1],
    'soy': [173, 17, 10, 9, 3],
    'spring onion': [32, 1.8, 7.3, 0.2, 2.3],
    'steak': [271, 25, 0, 19, 0],
    'strawberry': [32, 0.7, 7.7, 0.3, 4.9],
    'tea': [1, 0, 0.2, 0, 0],
    'tofu': [76, 8, 1.9, 4.8, 0.6],
    'tomato': [18, 0.9, 3.9, 0.2, 2.6],
    'walnut': [654, 15, 14, 65, 2.6],
    'watermelon': [30, 0.6, 7.6, 0.1, 6],
    'white button mushroom': [22, 3.1, 3.3, 0.3, 2],
    'white radish': [18, 0.6, 4.1, 0.1, 2.5],
    'wine': [83, 0.1, 2.7, 0, 0.8],
    'wonton dumplings': [200, 8, 25, 8, 1]
}

def populate_db(db_path='density_db.xlsx'):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    # Read the existing DB
    df = pd.read_excel(db_path, sheet_name='Density DB')
    
    # Columns to update
    nutri_cols = ['kcal', 'protein', 'carbs', 'fat', 'sugar']
    
    updates_count = 0
    
    # Iterate through the DataFrame and update rows
    for index, row in df.iterrows():
        food_name = row['Food']
        
        # Exact match logic or simple normalization
        # The provided dictionary looks like it matches the keys well.
        
        # We can try to look up directly
        if food_name in nutrition_map:
            vals = nutrition_map[food_name]
            df.at[index, 'kcal'] = vals[0]
            df.at[index, 'protein'] = vals[1]
            df.at[index, 'carbs'] = vals[2]
            df.at[index, 'fat'] = vals[3]
            df.at[index, 'sugar'] = vals[4]
            updates_count += 1
        else:
            print(f"Warning: No nutrition data found for '{food_name}'")
            
    # Save back
    df.to_excel(db_path, sheet_name='Density DB', index=False)
    print(f"Successfully updated {updates_count} items in {db_path}")

if __name__ == '__main__':
    populate_db()
