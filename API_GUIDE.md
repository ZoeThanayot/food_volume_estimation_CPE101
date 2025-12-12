# Food Volume Estimation API Guide

This guide explains how to host the API and how to let a friend access it from another computer.

## 🟢 For the Host (You)

### 1. Start the API Server

Open your terminal, navigate to the project folder, and run:

```bash
# Activate conda environment
conda activate food_volume_estimation

# Start the server
python food_volume_estimation_app.py \
    --depth_model_architecture models/depth_architecture.json \
    --depth_model_weights models/depth_weights.h5 \
    --segmentation_model_weights models/FoodSeg.onnx \
    --segmentation_model_config models/FoodSeg.yaml \
    --density_db_source density_db.xlsx \
    --classifier_model models/EfficientNet-B0.pth \
    --classifier_label_map models/Efficient_Label_Map.json
```
*Wait until you see:* `* Running on http://127.0.0.1:5001`

### 2. Expose to the Internet (ngrok)

Open a **new terminal window** and run:

```bash
ngrok http 5001
```

Copy the URL that looks like: `https://xxxx-xxxx.ngrok-free.app`
👉 **Send this URL to your friend.**

---

## 🔵 For the Client (Your Friend)

### 1. Install Requirements

You need Python installed. Then run:

```bash
pip install requests opencv-python numpy
```

### 2. Get the Client Script

You should have received the file `friend_client.py` (located in the `client/` folder of this project).

### 3. Run the Client

Use the URL provided by the host.

**Basic Usage:**
```bash
python friend_client.py --image "path/to/your_food.jpg" --url "https://intelligible-mobbishly-stefan.ngrok-free.dev/predict"
```

**Specifying Plate Size:**
If your plate is not standard (standard is 20cm or 0.2m), you can specify the diameter in meters:
```bash
# Example for a 25cm plate
python friend_client.py --image "food.jpg" --url "..." --plate 0.25
```

### 4. Interpretation
The script will output:
- **Food Type**: Detected food name.
- **Weight**: Estimated weight in grams.
- **Nutrition**: kcal, protein, carbs, fat, sugar.
