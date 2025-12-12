import requests
import json
import cv2
import numpy as np
import argparse
import base64

def test_api(image_path, url='http://localhost:5001/predict', plate_diameter=0.2):
    print(f"[*] Sending request to {url}...")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Encode image
    # The API expects a JSON with 'img' key containing the flattened array or similar.
    # Looking at app code:
    # content = request.get_json()
    # img_encoded = content['img']
    # img_byte_string = ' '.join([str(x) for x in img_encoded])
    # np_img = np.fromstring(img_byte_string, np.uint8, sep=' ')
    
    # It seems the API as written expects a list of numbers representing bytes?
    # Line 90: img_encoded = content['img']
    # Line 91: img_byte_string = ' '.join([str(x) for x in img_encoded]) 
    
    # Wait, passing a list of million integers via JSON is inefficient.
    # But let's follow the existing logic I saw in `food_volume_estimation_app.py`.
    # "The array of estimated volumes in JSON format."
    
    # Preparing the payload as the server expects it.
    # The server does: `img_byte_string = ' '.join([str(x) for x in img_encoded])`
    # Then `np.fromstring(..., sep=' ')`
    # So `img_encoded` should be a list of byte values (ints).
    
    _, img_encoded = cv2.imencode('.jpg', img)
    img_list = img_encoded.flatten().tolist()
    
    payload = {
        'img': img_list,
        'plate_diameter': plate_diameter
    }
    
    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error connecting to API: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Food Volume API')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--url', type=str, default='http://localhost:5001/predict', help='API URL')
    parser.add_argument('--plate_diameter', type=float, default=0.2, help='Expected plate diameter in meters (default: 0.2)')
    
    args = parser.parse_args()
    test_api(args.image, args.url, args.plate_diameter)
