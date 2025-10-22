# Resume Screening Application Setup Guide

This guide provides step-by-step instructions to set up and run the Resume Screening application using both `uv` and `conda` environments. Follow the steps below based on your preferred environment setup.

---

## 1️⃣ Using `uv` for Environment Management

### Step 1: Install `uv`
Ensure `uv` is installed on your system. If not, install it using pip:
```bash
pip install uv
```

### Step 2: Create and Activate the Environment
Create a new environment named `resume-screening`:
```bash
uv env create resume-screening
```
Activate the environment:
```bash
uv activate resume-screening
```

### Step 3: Install Dependencies
Install all dependencies listed in `pyproject.toml`:
```bash
poetry install --no-root
```

### Step 4: Run the Application
Run the Streamlit application:
```bash
uv run streamlit run streamlit_app.py
```

---

## 2️⃣ Using `conda` for Environment Management

### Step 1: Install `conda`
Ensure `conda` is installed on your system. If not, download and install it from the [official website](https://docs.conda.io/en/latest/miniconda.html).

### Step 2: Create and Activate the Environment
Create a new environment named `Resume_screening` with Python 3.11:
```bash
conda create -n Resume_screening python=3.11
```
Activate the environment:
```bash
conda activate Resume_screening
```

### Step 3: Install Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

---

## Additional Notes

### Generating `requirements.txt`
If you need to generate a `requirements.txt` file for deployment or sharing:
```bash
pip freeze > requirements.txt
```

### Exporting Conda Environment
To save your `conda` environment for reproducibility:
```bash
conda env export > environment.yml
```
To recreate the environment on another machine:
```bash
conda env create -f environment.yml
```

### Example `requirements.txt`
```plaintext
chromadb==0.3.23
docx2txt==0.8
google-generativeai==0.1.2
langchain==0.1.78
python-dotenv==1.0.0
streamlit==1.25.0
uvicorn==0.22.0
```

---

By following these steps, you can set up and run the Resume Screening application using either `uv` or `conda`. Choose the method that best suits your workflow.