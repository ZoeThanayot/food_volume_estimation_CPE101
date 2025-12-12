import requests
import json
import cv2
import argparse
import sys

def predict_food(image_path, api_url, plate_diameter=0.2):
    """
    Sends an image to the Food Volume Estimation API.
    
    Args:
        image_path (str): Path to the image file.
        api_url (str): public URL of the API (e.g., ngrok URL).
        plate_diameter (float): Expected plate diameter in meters (default 0.2m = 20cm).
    """
    print(f"[*] Reading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Encode image to jpg
    _, img_encoded = cv2.imencode('.jpg', img)
    img_list = img_encoded.flatten().tolist()
    
    payload = {
        'img': img_list,
        'plate_diameter': plate_diameter
    }
    
    print(f"[*] Sending request to {api_url}...")
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("\n---------- RESULT ----------")
            data = response.json()
            
            # Print Items
            print(f"Directory: {data.get('items', [])}")
            for item in data.get('items', []):
                print(f"\nFood: {item['food_type']}")
                print(f"Weight: {item['weight_g']} g")
                print(f"Calories: {item['kcal']} kcal")
                print(f"Protein:  {item['protein']} g")
                print(f"Carbs:    {item['carbs']} g")
                print(f"Fat:      {item['fat']} g")
                print(f"Sugar:    {item['sugar']} g")
            
            print("\n----------------------------")
            print(f"Total Weight: {data['total_weight_g']} g")
            total_nutri = data.get('total_nutrition', {})
            print(f"Total Calories: {total_nutri.get('kcal', 0)} kcal")
            print("----------------------------\n")
            
        else:
            print(f"Failed! Status Code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to API: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Friend Food API Client')
    parser.add_argument('--image', type=str, required=True, help='Path to your food image')
    parser.add_argument('--url', type=str, required=True, help='The API URL your friend gave you (e.g., https://xxxx.ngrok.io/predict)')
    parser.add_argument('--plate', type=float, default=0.2, help='Plate diameter in meters (default 0.2)')
    
    args = parser.parse_args()
    predict_food(args.image, args.url, args.plate)
