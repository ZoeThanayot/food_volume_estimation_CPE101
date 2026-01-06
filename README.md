# Food Volume & Nutrition Estimation API

A computer vision-based API that estimates the **volume**, **weight**, and **nutritional content** (Calories, Protein, Carbs, Fat, Sugar) of food from a single image. It also includes an **EfficientNet-B0** classifier to automatically detect the food type.

## 🌟 Features

- **Food Classification**: Detects 100+ Thai food menu items using EfficientNet-B0.
- **Volume Estimation**: Uses Monocular Depth Estimation to calculate food volume/weight.
- **Nutrition Calculation**: Maps food weight to a nutrition database (`density_db.xlsx`).
- **REST API**: Simple Flask-based API for easy integration.

---

## 🛠️ Installation

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Model Setup

You need to download the pre-trained models and place them in the `models/` directory.

| Model File                        | Description                              |
| :-------------------------------- | :--------------------------------------- |
| `models/depth_architecture.json`  | Depth Estimation Model Architecture      |
| `models/depth_weights.h5`         | Depth Estimation Model Weights           |
| `models/FoodSeg.onnx`             | Segmentation Model (Modified best2.onnx) |
| `models/FoodSeg.yaml`             | Segmentation Config                      |
| `models/EfficientNet-B0.pth`      | Food Classifier Weights                  |
| `models/Efficient_Label_Map.json` | Classifier Label Map                     |

_(Note: These files are large and usually not included in the git repo. Please obtain them from the project drive folder if they are missing.)_

---

## 🧠 Model Training

We have provided the training notebooks for our models on Kaggle:

- **Classification Model**: [Thai Food Classification](https://www.kaggle.com/code/zoethanayot/model-classify-food-thai)
- **Segmentation Model**: [Food Segmentation Training v2](https://www.kaggle.com/code/zoethanayot/foodseg-segmentation-trainingv2)

---

## 🚀 Usage

### 1. Start the API Server

Run the following command to start the server:

```bash
python food_volume_estimation_app.py \
    --depth_model_architecture models/depth_architecture.json \
    --depth_model_weights models/depth_weights.h5 \
    --segmentation_model_weights models/FoodSeg.onnx \
    --segmentation_model_config models/FoodSeg.yaml \
    --density_db_source density_db.xlsx \
    --classifier_model models/EfficientNet-B0.pth \
    --classifier_label_map models/Efficient_Label_Map.json
```

The server will start at `http://0.0.0.0:5001`.

### 2. Client Examples

#### Python Client (`client/friend_client.py`)

This script helps you easily send an image to the API.

```bash
# Basic Usage
python client/friend_client.py --image "assets/rice.jpg" --url "http://localhost:5001/predict"

# With Custom Plate Size (Default is 0.2m or 20cm)
python client/friend_client.py --image "assets/rice.jpg" --url "http://localhost:5001/predict" --plate 0.25
```

#### Classify Endpoint

To just check what food it is:

```bash
python scripts/test_classify.py --image "assets/rice.jpg"
```

---

## 🌐 Remote Access (ngrok)

To allow friends to access your local API over the internet:

1.  Start the server (Step 1 above).
2.  Install and run **ngrok**:
    ```bash
    ngrok http 5001
    ```
3.  Send the `https://...ngrok-free.app` URL to your friend.
4.  They can use the client script with that URL.

---

## 📂 Project Structure

- `food_volume_estimation_app.py`: Main API server.
- `density_db.xlsx`: Database containing food density and nutrition info.
- `client/`: Contains `friend_client.py` for easy testing.
- `scripts/`: Helper scripts (downloading models, testing).
- `models/`: Stores model weights (ignored by git).
- `dev_archive/`: (Ignored) Old development scripts.

---

**Enjoy your Food AI!** 🍛🤖

## 🙏 Acknowledgements

This project is built upon the work of **AlexGraikos**.

- Original Repository: [food_volume_estimation](https://github.com/AlexGraikos/food_volume_estimation)

Special thanks to the open-source community for the models and tools used in this project.
