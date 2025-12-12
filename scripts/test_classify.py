import requests
import cv2
import argparse
import sys
import json

def test_classify(image_path, url='http://localhost:5001/classify'):
    print(f"[*] Reading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Encode image to jpg (API expects flat list of bytes/ints from encoded img)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_list = img_encoded.flatten().tolist()
    
    payload = {
        'img': img_list
    }
    
    print(f"[*] Sending request to {url}...")
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("\n---------- CLASSIFICATION RESULT ----------")
            data = response.json()
            print(json.dumps(data, indent=4, ensure_ascii=False))
            print("-------------------------------------------\n")
        else:
            print(f"Failed! Status Code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to API: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Food Classification API')
    parser.add_argument('--image', type=str, required=True, help='Path to food image')
    parser.add_argument('--url', type=str, default='http://localhost:5001/classify', help='API URL')
    
    args = parser.parse_args()
    test_classify(args.image, args.url)
