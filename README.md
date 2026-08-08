# Fake Job Posting Detection System

A Django-based web application that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to identify potentially fraudulent job postings from job descriptions and related metadata.

The system provides separate **User** and **Admin** modules, allowing users to submit job posting information for prediction while administrators can manage users and view machine learning classification reports.

---

## Table of Contents

* [Overview](#-overview)
* [Project Highlights](#-project-highlights)
* [Key Features](#-key-features)
* [Technology Stack](#-technology-stack)
* [Machine Learning Pipeline](#-machine-learning-pipeline)
* [Machine Learning Models](#-machine-learning-models)
* [Project Structure](#-project-structure)
* [Prerequisites](#-prerequisites)
* [Installation & Setup](#-installation--setup)
* [Usage & Workflow](#-usage--workflow)
* [Dataset](#-dataset)
* [Future Enhancements](#-future-enhancements)
* [License](#-license)
* [Author](#-author)

---

## Overview

Fraudulent job postings are increasingly used to deceive job seekers, collect personal information, and conduct employment-related scams.

This project aims to provide an automated solution for identifying potentially fraudulent job advertisements using **Machine Learning and Natural Language Processing**.

The application analyzes multiple attributes of a job posting, including:

* Job title
* Location
* Company profile
* Job description
* Requirements
* Benefits
* Industry

The processed information is converted into numerical features using **Bag-of-Words (`CountVectorizer`)** and passed through multiple machine learning classification algorithms.

The system ultimately classifies a job posting as:

> 🟢 **Real Job**
> 🔴 **Fraudulent Job**

---

## Project Highlights

*  Django-based full-stack web application
*  Machine Learning-based fake job classification
*  NLP-based text preprocessing
*  Comparison of multiple ML classification algorithms
*  Class imbalance handling using `RandomUnderSampler`
*  User registration and authentication
*  Administrator user-activation system
*  Classification reports and model performance analysis
*  Real-time job posting prediction through a web interface
*  Dataset exploration and preprocessed-data inspection

---

##  Key Features

###  User Module

#### Registration & Authentication

* User registration through a Django form.
* Administrator approval/activation workflow.
* Login system for activated users.

#### Dataset Viewer

* Allows users to inspect sample job posting records used by the system.

#### Preprocessed Data

* Displays cleaned and processed text data used for machine learning.

#### ML Classification Reports

* Provides performance information for multiple machine learning models.
* Includes metrics such as:

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix

#### Live Prediction

Users can enter job posting information through an interactive form and receive an automated prediction:

```text
Real Job
     or
Fraudulent Job
```

---

###  Admin Module

#### Admin Authentication

Dedicated administrator login functionality.

#### User Activation

Administrators can:

* View registered users.
* Review user registrations.
* Activate users.
* Control access to the application.

#### Classification Analytics

Administrators can view machine learning classification results and compare the performance of different models.

---

##  Technology Stack

| Category                     | Technologies                   |
| ---------------------------- | ------------------------------ |
| **Programming Language**     | Python                         |
| **Backend Framework**        | Django 4.0+                    |
| **Frontend**                 | HTML5, CSS3, JavaScript        |
| **Template Engine**          | Django Templates               |
| **Database**                 | SQLite3                        |
| **Data Processing**          | Pandas, NumPy                  |
| **NLP**                      | NLTK                           |
| **Feature Extraction**       | Scikit-Learn `CountVectorizer` |
| **Machine Learning**         | Scikit-Learn                   |
| **Imbalanced Data Handling** | Imbalanced-Learn               |
| **Visualization**            | Matplotlib, Seaborn, WordCloud |
| **Image Processing**         | Pillow                         |
| **Version Control**          | Git & GitHub                   |
| **Development Environment**  | Visual Studio Code             |

---

##  Machine Learning Pipeline

The system follows the following machine learning workflow:

```text
┌──────────────────────┐
│   Raw Job Dataset    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Data Cleaning        │
│ & Preprocessing      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Handle Class         │
│ Imbalance            │
│ RandomUnderSampler   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Text Concatenation   │
│ of Job Attributes    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ NLTK Stopword        │
│ Removal              │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ CountVectorizer      │
│ Bag-of-Words         │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Machine Learning     │
│ Model Training       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Real-Time Prediction │
└──────────────────────┘
```

### 1. Feature Engineering

The following job-posting attributes are combined for analysis:

```text
title
location
company_profile
description
requirements
benefits
industry
```

### 2. Data Preprocessing

The text data undergoes preprocessing steps such as:

* Lowercasing
* Punctuation removal
* Stop-word removal
* Text cleaning
* Feature extraction

English stopwords are handled using:

```python
nltk.corpus.stopwords
```

### 3. Handling Class Imbalance

The dataset may contain significantly more legitimate job postings than fraudulent postings.

To address this imbalance, the project uses:

```python
RandomUnderSampler
```

from the `imbalanced-learn` library.

### 4. Feature Extraction

The cleaned text is transformed into numerical feature vectors using:

```python
CountVectorizer
```

This implements a **Bag-of-Words** representation of the job posting text.

---

##  Machine Learning Models

The project compares multiple classification algorithms:

| Model                       | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| **Multinomial Naive Bayes** | Probabilistic text classification        |
| **K-Nearest Neighbors**     | Distance-based classification            |
| **Decision Tree**           | Rule-based classification                |
| **Random Forest**           | Ensemble tree-based classification       |
| **Support Vector Machine**  | Classification using decision boundaries |
| **MLP Classifier**          | Neural-network-based classification      |

The application provides classification metrics to compare the performance of these models.

---

##  Project Structure

```text
FakeJobPosting/
│
├── .gitignore
├── README.md
├── manage.py
│
├── FakeJobPosting/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── views.py
│   ├── wsgi.py
│   └── asgi.py
│
├── admins/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
│   └── migrations/
│
├── users/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
│   ├── migrations/
│   └── utility/
│       ├── PreprocessedData.py
│       └── predections.py
│
├── assets/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── fonts/
│   │   └── images/
│   │
│   └── templates/
│       ├── AdminLogin.html
│       ├── UserLogin.html
│       ├── UserRegistrations.html
│       ├── base.html
│       ├── index.html
│       ├── admins/
│       └── users/
│
└── media/
    └── DataSet.csv    # Local dataset, excluded from GitHub
```

> **Note:** `db.sqlite3` and `media/DataSet.csv` are excluded from the GitHub repository using `.gitignore`. The dataset is approximately **55 MB**, so it is not stored directly in the repository.

---

##  Prerequisites

Before running the project, make sure you have:

* **Python 3.8 or later**
* **pip**
* **Git**
* A modern web browser

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/BougiSwarna/FakeJobPosting.git
cd FakeJobPosting
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install django pandas numpy scikit-learn imbalanced-learn nltk wordcloud matplotlib seaborn pillow
```

### 4. Download NLTK Resources

Run:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 5. Add the Dataset

The original dataset is not included in this GitHub repository because of its file size.

Place the required dataset locally at:

```text
media/DataSet.csv
```

### 6. Apply Database Migrations

Run:

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Start the Development Server

Run:

```bash
python manage.py runserver
```

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

---

##  Usage & Workflow

###  Administrator Workflow

1. Open the Admin Login page.
2. Log in using administrator credentials.
3. Open the registered-users section.
4. Review newly registered users.
5. Activate users who should receive access.
6. View classification reports and model performance.

Admin login:

```text
http://127.0.0.1:8000/AdminLogin/
```

---

###  User Workflow

1. Register a new account.
2. Wait for administrator activation.
3. Log in using the activated account.
4. Explore the available features:

   * View Dataset
   * View Preprocessed Data
   * View ML Reports
   * Test Job Posting
5. Enter job posting information.
6. Submit the information for prediction.
7. View the classification result:

```text
Real Job
or
Fraudulent Job
```

---

## Dataset

The project uses a dataset containing job-posting information and labels indicating whether a posting is legitimate or fraudulent.

The dataset contains attributes such as:

```text
title
location
company_profile
description
requirements
benefits
industry
```

### Dataset Availability

The dataset is **not included in this GitHub repository** because the file is approximately **55 MB**, exceeding GitHub's recommended maximum file size for individual files.

For local execution, place the dataset at:

```text
media/DataSet.csv
```

---

##  Default Admin Credentials

For demonstration purposes, the application may use default administrator credentials configured within the project.

If your local project uses:

```text
Username: admin
Password: admin
```

these should be treated as **demo credentials only**.

> ⚠️ Do not use default credentials in a production environment. Change them before deployment.

---

##  Future Enhancements

Potential improvements for future versions include:

*  Secure password hashing and improved authentication
*  Deployment to a cloud platform
*  Migration from SQLite to PostgreSQL/MySQL
*  Integration of advanced NLP models such as TF-IDF, Word2Vec, BERT, or transformer-based models
*  Interactive model-performance dashboards
*  Explainable AI for prediction results
*  Responsive UI improvements
*  REST API for external job-posting analysis
*  Automated unit and integration testing
*  Automated model retraining with new datasets
*  Advanced fraud-pattern analytics

---

##  License

This project is intended for **educational, research, and demonstration purposes**.

---

## Author

**Bougi Swarna**

B.Tech – Computer Science and Engineering

GitHub:
https://github.com/BougiSwarna

---

 If you find this project useful, consider giving the repository a **star** on GitHub!
