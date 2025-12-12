import gdown
import os

def download_models():
    # Create models directory if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs('../models')

    # Depth prediction model
    print("Downloading Depth prediction model...")
    # Architecture
    gdown.download(id='1t-nlUvbtD6ungcj0I_BweubRnbUogjVG', output='../models/depth_architecture.json', quiet=False)
    # Weights
    gdown.download(id='1fbzlVq3KaqVBtsLNBEsEMQLnr-hc8GkB', output='../models/depth_weights.h5', quiet=False)

    # Segmentation model
    print("Downloading Segmentation model...")
    # Weights
    gdown.download(id='1Y-TEatoy16QwHyRQvWBkvd0ZHd4E0Lgd', output='../models/segmentation_weights.h5', quiet=False)

    # Density Database
    print("Downloading Density Database...")
    url = 'http://www.fao.org/fileadmin/templates/food_composition/documents/density_DB_v2_0_final-1__1_.xlsx'
    # We can use requests or wget, but gdown is for drive. 
    # Let's use urllib since it is standard library
    import urllib.request
    urllib.request.urlretrieve(url, '../density_db.xlsx')

    print("Download complete.")

if __name__ == "__main__":
    download_models()
