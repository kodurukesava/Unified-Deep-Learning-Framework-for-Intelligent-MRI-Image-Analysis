# Unified Deep Learning Framework for Intelligent MRI Image Analysis

## About the Project

The Unified Deep Learning Framework for Intelligent MRI Image Analysis is a web-based application developed to analyse brain MRI images and identify different types of brain tumors using deep learning techniques.

The application allows users to upload brain MRI images and obtain prediction results through a simple web interface. Once an image is uploaded, the trained deep learning model processes the image and predicts its category.

In addition to MRI image classification, the system includes a Doctor/Admin Dashboard to manage patient information, scan details, and prediction history. It also provides PDF report generation for storing and viewing MRI analysis results.

The main objective of this project is to combine deep learning with web technology to create an accessible and easy-to-use MRI image analysis system.

## Features

- **MRI Image Upload:** Allows users to upload brain MRI images.
- **Tumor Classification:** Uses a trained deep learning model to classify MRI images.
- **Prediction Results:** Displays the detected tumor category.
- **Doctor/Admin Dashboard:** Provides access to patient information and MRI analysis records.
- **Patient Management:** Maintains patient details and scan information.
- **Prediction History:** Stores previous MRI prediction records.
- **PDF Report Generation:** Generates downloadable reports containing prediction details.
- **User-Friendly Interface:** Provides a simple interface developed using HTML and CSS.
- **Database Integration:** Stores application and prediction information using SQLite.

## Tumor Categories

The system is designed to classify brain MRI images into four categories:

1. Glioma Tumor
2. Meningioma Tumor
3. Pituitary Tumor
4. No Tumor

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application development |
| Deep Learning | MRI image classification |
| TensorFlow / Keras | Model training and prediction |
| HTML | Web page structure |
| CSS | Frontend design |
| SQLite | Database management |
| NumPy | Numerical operations |
| Pillow / OpenCV | Image processing |

## Project Structure

```text
Unified-Deep-Learning-Framework/
│
├── app.py
├── model/
│   └── trained_model.h5
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── login.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── images/
│   └── uploads/
│
├── database/
│   └── mri_database.db
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: This is a sample structure. The actual folders and filenames may differ depending on the implementation.*

## Installation and Setup

Follow the steps below to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### Step 2: Navigate to the Project Directory

```bash
cd your-repository-name
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Linux / macOS:

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python app.py
```

### Step 7: Open the Application

Open your browser and visit:

```text
https://mri-analysis-899514544884.asia-south1.run.app
```

The MRI Image Analysis application will be available locally.

## Working Procedure

The application follows the workflow below:

1. The user opens the MRI Image Analysis web application.
2. A brain MRI image is uploaded through the interface.
3. The application processes the uploaded image.
4. The trained deep learning model analyses the image.
5. The predicted tumor category is displayed.
6. Patient and prediction details are stored in the SQLite database.
7. The Doctor/Admin can access patient records and previous prediction details through the dashboard.
8. A PDF report can be generated based on the prediction results.

## Database Management

The project uses SQLite as its database management system.

The database is used to maintain:

- Patient information.
- MRI scan details.
- Prediction results.
- Previous prediction records.
- Related application information.

SQLite provides a lightweight database solution that can be integrated directly with the Flask application.

## Team Members and Contributions

This project was developed collaboratively by a team of four members. Each member contributed to different stages of the development process, including model development, image processing, web application development, database integration, and testing.


### Acknowledgement

We would like to thank all our team members for their valuable contributions and cooperation throughout the development of this project. Working together helped us gain practical knowledge of deep learning, medical image processing, web application development, and database management.

## Screenshots

Screenshots of the application can be added here to demonstrate its working.

### Home Page

<img width="339" height="488" alt="image" src="https://github.com/user-attachments/assets/c363c6ee-585f-4cb4-b3b2-b4fbeecc98b4" />


### MRI Prediction Result

<img width="270" height="457" alt="image" src="https://github.com/user-attachments/assets/18fe2096-702f-45dd-bdf2-392ca5adf13f" />


### Doctor/Admin Dashboard

<img width="959" height="497" alt="image" src="https://github.com/user-attachments/assets/42fa169c-77d2-4ad5-9236-1bd66436de2b" />


### PDF Report

<img width="368" height="352" alt="image" src="https://github.com/user-attachments/assets/862a32d8-e34d-4459-9b68-905e1b3af788" />


## Future Enhancements

The project can be improved further with the following features:

- Integration of more advanced deep learning architectures.
- Improvement in classification accuracy using larger MRI datasets.
- Addition of MRI tumor segmentation.
- Comparison of previous MRI scan results.
- Improved patient record management.
- Cloud-based storage and deployment.
- Enhanced authentication and security features.

## Disclaimer

This project is developed for academic and educational purposes. The predictions generated by the system are intended for research and demonstration only. The results should not be considered a substitute for professional medical diagnosis, clinical evaluation, or advice from qualified healthcare professionals.

## Project Information

**Project Title:** Unified Deep Learning Framework for Intelligent MRI Image Analysis

**Domain:** Deep Learning and Medical Image Processing

**Application Type:** Web-Based Application

**Backend:** Python Flask

**Database:** SQLite

**Frontend:** HTML and CSS

**Project Team Size:** 4 Members

## License

This project is intended for educational and academic research purposes.
