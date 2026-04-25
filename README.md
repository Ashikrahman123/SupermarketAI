# Supermarket AI - Product Image Classification

## About This Project

This is a college-level academic project that classifies supermarket product images into simple categories and shows an age-group hint for the predicted category.

The project is designed for learning and demonstration, not enterprise production.

## What This Project Does

1. Trains an image classifier using transfer learning (MobileNetV2).
2. Predicts product category from an uploaded image.
3. Uses hybrid inference:
   model prediction + dataset-similarity matching.
4. Displays confidence, reliability, and class probability breakdown.
5. Provides a Streamlit UI with Analyzer + Interactive Dashboard.
6. Provides a basic FastAPI endpoint for prediction.
7. Includes a behavior-segmentation module using Hierarchical Clustering in dashboard analytics.
8. Includes a Profit/Loss Forecast module based on business inputs (unit cost, sale price, quantity).

## Categories

Current dataset classes:

1. Baby products
2. bakery
3. Beauty
4. electronics
5. Grocery
6. household
7. Snacks
8. Stationaries
9. Toys

Notes:

1. Class names are discovered dynamically from folder names under dataset/.
2. When new folders are added and training is rerun, metadata updates automatically.
3. Unmapped classes in age metadata default to All ages.

## Tech Stack

1. Python
2. TensorFlow / Keras
3. Streamlit
4. FastAPI
5. NumPy, Pillow

## Project Structure

1. app.py - Main Streamlit app (Analyzer + Dashboard)
2. train.py - Model training script
3. predict.py - CLI prediction script
4. hybrid_inference.py - Similarity-based hybrid inference utilities
5. model.keras / model_metadata.json - Trained artifacts
6. src/core/ - Shared config and inference engine
7. src/api/ - FastAPI app and response schemas
8. dataset/ - Class-wise training images

## How Inference Works

1. The model predicts probabilities with test-time augmentation (TTA).
2. A similarity score is computed against class centroids from dataset images.
3. Both are blended with a hybrid weight (from metadata).
4. Final output includes:
   category, age group, confidence, uncertainty flag, and top-2 margin.

## Profit/Loss Forecast Module

This module is a business layer built on top of image prediction.

Why this design:

1. Image model predicts product category and confidence.
2. Profit and loss depend on business inputs, not image pixels.
3. So the app combines prediction + user-entered transaction inputs.

Inputs:

1. Unit Cost
2. Sale Price
3. Quantity Sold

Computed outputs:

1. Revenue = sale_price \* quantity
2. Total Cost = unit_cost \* quantity
3. Profit/Loss = revenue - total_cost
4. Profit Margin (%) = (profit / revenue) \* 100
5. Expected Profit (rule-based forecast)

Rule-based forecast:

1. Each category has a base expected margin rate.
2. Confidence and uncertainty adjust this expected margin.
3. Expected Profit = revenue \* expected_margin_rate

This is intentionally simple and explainable for college evaluation.

## Setup

1. Create environment and install:

```bash
make install
```

2. Train model:

```bash
make train
```

Training safety behavior:

1. The script validates dataset/ before training.
2. It fails fast if a class folder exists but contains no valid images.
3. It writes model_metadata.json with discovered class_names and class_counts_train.

4. Run Streamlit app:

```bash
make run
```

4. Run API:

```bash
make api
```

Profit/Loss API example:

```bash
curl -X POST http://127.0.0.1:8000/predict-business \
   -F "file=@dataset/Snacks/SNAC 1.jpeg" \
   -F "unit_cost=20" \
   -F "sale_price=32" \
   -F "quantity=3"
```

5. Run CLI prediction:

```bash
python predict.py "dataset/Baby products/BP 5.jpeg"
```

## Dashboard Features

1. Prediction history (session-based)
2. Confidence trend visualization
3. Category distribution chart
4. Reliability mix chart
5. CSV and JSON export
6. Clear history action
7. Behavioral segmentation view (cluster scatter + operational summary)

## Behavioral Segmentation with Hierarchical Clustering

The dashboard includes an operational segmentation layer based on
Agglomerative Hierarchical Clustering over recent prediction history.

Input signals used for clustering:

1. Confidence
2. Top-2 margin
3. Uncertain flag (numeric)

Pipeline:

1. Min-Max scaling
2. Agglomerative Hierarchical Clustering (Ward linkage)
3. Cluster scatter visualization
4. Segment summary table (count, average confidence, average margin, uncertain rate)

Use case:
This module helps explain model behavior in a structured, business-friendly way by grouping similar prediction outcomes into interpretable segments.

## Scope Note

This repository is an academic demo. Keep implementation understandable for project viva and review.
Avoid overcomplicated production architecture.
