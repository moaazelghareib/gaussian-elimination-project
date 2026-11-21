# 🧮 Step-by-Step Linear System Solver

An interactive web application built with **Python** and **Streamlit** to solve systems of linear equations using **Gaussian Elimination** and **Gauss-Jordan Elimination**. The app provides detailed step-by-step elementary row operations formatted in **LaTeX**, supports exact fractional arithmetic, and detects unique, inconsistent (no solution), and infinite-solution systems.

---

## ✨ Features

- **Dual Elimination Methods:**
  - **Gaussian Elimination:** Reduces the augmented matrix $[A|b]$ to **Row Echelon Form (REF)** and performs back substitution.
  - **Gauss-Jordan Elimination:** Reduces the augmented matrix to **Reduced Row Echelon Form (RREF)**.
- **Dynamic $n$-Variable & $m$-Equation Grid:**
  - Configure the number of variables ($1$ to $15$) and equations ($m \le n$).
  - Clean coefficient grid input supporting integers and exact fractions (e.g., `1/2`, `-3/4`).
- **Complete Solution Classification:**
  - **Unique Solution:** Displays exact values for $X_1, X_2, \dots, X_n$.
  - **No Solution:** Detects inconsistent systems ($0 = c$ where $c \neq 0$).
  - **Infinite Solutions:** Identifies free variables and expresses basic variables parametrically.
- **Image-to-Equation OCR (Offline):**
  - Upload an image of printed/clear linear equations (`JPG`, `PNG`) using **OpenCV** preprocessing and **Tesseract OCR**, parsed automatically via **SymPy**.
- **Academic LaTeX Visualization:**
  - Displays every row swap, scaling, and elimination step with clear matrix notation.

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Symbolic & Exact Math:** `fractions.Fraction`, [SymPy](https://www.sympy.org/)
- **Image Processing & OCR:** [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/), [PyTesseract](https://pypi.org/project/pytesseract/), [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## 📂 Project Structure

```text
├── gaussian solver project_2.py   # Main Streamlit application
├── requirements.txt               # Python dependencies
├── packages.txt                   # System-level packages (Tesseract OCR for Linux/Cloud)
├── .gitignore                     # Ignored files and virtual environments
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Install Tesseract OCR (Required for Image Upload Tab)
- **Windows:**
  1. Download and install Tesseract OCR from [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
  2. Ensure it is installed at the default path:
     ```text
     C:\Program Files\Tesseract-OCR\tesseract.exe
     ```
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr
  ```
- **macOS:**
  ```bash
  brew install tesseract
  ```

### 4. Run the Application
```bash
streamlit run "gaussian solver project_2.py"
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

1. **Coefficients Grid Tab:**
   - Select the **Number of Variables**, **Number of Equations**, and **Elimination Method**.
   - Enter the coefficients for each variable ($X_1, X_2, \dots$) and the right-hand side (**RHS**) constant.
   - Click **Solve System** to view every matrix transformation step and the final result.
2. **Image Upload Tab:**
   - Choose your preferred elimination method and number of variables.
   - Upload a clear image of your linear system (`JPG` or `PNG`) to extract and solve the equations automatically.