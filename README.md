# Fall Safe: Real-Time Fall Detection System

## Abstract

**Fall Safe** is designed to address fall-related injuries among vulnerable populations by leveraging computer vision and machine learning. The system detects falls in real-time from CCTV footage, analyzing video streams to identify abnormal movements and postures. Alerts are sent to caregivers or emergency services with details about the incident, aiming to improve response times and safety for at-risk individuals.

## Features

- **Real-Time Fall Detection**: Utilizes YOLO11 for accurate fall detection.
- **Integration**: Works with existing CCTV setups.
- **Alerts**: Sends notifications with incident details to caregivers or emergency services.

## Getting Started

### Prerequisites

- **Python**: Latest version
- **NVIDIA GPU** (highly recommended): For accelerated processing

### Setup and Installation

1. **Install Python**

   - Download and install from [Python's official website](https://www.python.org/downloads/).

2. **Set Up YOLO11 Project**

   ```bash
   mkdir YOLO_PROJECT/yolo11-python
   cd YOLO_PROJECT/yolo11-python
   ```

3. **Create a Virtual Environment**

   - **Using venv**:

     ```bash
     python -m venv env
     ```

     Activate the virtual environment:

     - On Windows:
       ```bash
       .\env\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source env/bin/activate
       ```

   - **Using conda**: For detailed instructions on creating a conda environment, refer to the [Official Anaconda Documentation](https://docs.anaconda.com/anaconda/install/).

4. **Install GPU Drivers and CUDA**

   - Install NVIDIA GPU drivers.

5. **Install Required Packages**

   - Install the packages from the "requirements.txt" file.

   ```bash
   pip install -r requirements.txt
   ```

   - Install pyTorch and its dependencies. Get the architecture, platform and select the latest version of cuda from [pyTorch Website](https://pytorch.org/)
     It should be like this for Windows

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

6. **Verification**

   - Verify all installations by running the test files from the "TestFiles" folder.
     - Run "Test_Camera.py" to get the ids for the camera source.
     - Run "Test_Cuda_GPU.py" to check the proper verification of cuda installation and gpu detection.

### Running the System

1. **Test YOLO11 Inference according to Ultralytics website and detection file**

   ```bash
   python detection.py --model yolo11n.onnx --source data/images/horses.jpg
   python detection.py --model yolo11n.onnx --source data/videos/road.mp4
   python detection.py --model yolo11n.onnx --source 0
   ```

   \*NOTE: Get the detection.py from ultralytics github page and for yolo11

2. **Prepare and Get Labelled Dataset from Roboflow**

   - Upload images to Roboflow and label them as either `fall` or `nofall`.
   - Discard any images that are not relevant by marking them as null.
   - Download the structured dataset from Roboflow and select YOLO11 for model type when prompted.
   - Extract the zip file from roboflow

3. **Train the Model**

   - Modify the name for the current operation.
   - Adjust the parameters value to properly utilize the GPU.

   **_Training through CLI_** :

   ```bash
   yolo detect train model=yolo11n.pt data="path/to/dataset" imgsz=224 device=0 workers=2 batch=16 epochs=100 patience=50 name=yolo11_fallsafe_detection
   ```

   **_Training through GUI_** :

   - Use python for training the model.
   - Use Roboflow website for training.
   - Use Ultralyticss HUB.
   - Use Google Colab for training

4. **Continue Training after Pause OR Further Train model with new/updated Dataset**

   **_Training through CLI_** :

   ```bash
   yolo detect train model=runs/classify/yolo11_fallsafe_detection/weights/last.pt resume=True
   ```

   **_Training through GUI_** : Restart Training as most services dont support pausing.

5. **Perform Classification**

   ```bash
   yolo detect predict model=runs/classify/yolo11_fallsafe_detection/weights/best.pt source="path/image.jpg" save=True
   ```

6. **Real-Time Classification via Camera**
   ```bash
   yolo detect predict model=runs/classify/yolo11_fallsafe_detection/weights/best.pt source="0" save=True conf=0.5 show=True save_txt=True line_thickness=1
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact us at Issues Pages.

## Authors

- [Syed Arbaaz Hussain](https://github.com/SyedArbaazHussain)
- [Adithi N Gatty](https://github.com/AdithiNgatty)
- [Prabuddh Shetty](https://github.com/Prabuddhshetty901)
- [Shreya S Rao](https://github.com/shreyarao515)

---

**Fall Safe** is developed by the above contributors. For more information, visit [our GitHub repository](https://github.com/FallSafe).
