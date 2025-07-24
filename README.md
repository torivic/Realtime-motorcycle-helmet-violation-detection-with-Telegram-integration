# Realtime-motorcycle-helmet-violation-detection-with-Telegram-integration

This project implements a real-time motorcycle helmet violation detection system using YOLOv8s. It can process video input from a file or selected camera device to detect motorcycles and whether riders are wearing helmets. When a rider without a helmet is detected, the system retrieves the location via Chrome/Selenium and sends a notification with a picture of the motorcycle and its license plate number to your phone using a Telegram bot created with @BotFather.

---

## Dataset

The dataset used for training is sourced from [Roboflow](https://universe.roboflow.com/oriorio/violation-detection-t5bf9/dataset/1), which was uploaded and labeled by myself. It contains annotations for:

- Helmets (`helm`)
- Motorcycles (`motor`)
- Riders without helmets (`tanpa helm`)
- License plates (`plat`)

---

## Features

- Real-time detection from selected camera device or video file
- Detects motorcycles, helmet usage, and license plates  
- Retrieves geographic location automatically via Selenium  
- Sends instant Telegram notifications with images of violations (including motorcycle and plate)  
- Built on the lightweight and accurate YOLOv8s model  
