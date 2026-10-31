import streamlit as st
from fractions import Fraction
import re
from PIL import Image
from sympy import symbols, Eq, linear_eq_to_matrix
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import cv2
import numpy as np
import sys
import pytesseract

# لو شغالين على جهازك (ويندوز) هيستخدم المسار ده، لو على السيرفر هيعتمد على مسار لينكس التلقائي
if sys.platform == 'win32':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------------------------------------------------
# 1. إعداد واجهة الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="Linear Algebra Solver", layout="wide")
st.title("🧮 Step-by-Step Gaussian Elimination Solver")
st.write("ارفع صورة المعادلات (يفضل مطبوعة أو بخط واضح) وسيتم حلها خطوة بخطوة.")

# ---------------------------------------------------------
# 2. استخراج المعادلات عبر Tesseract OCR (أوفلاين بالكامل)
# ---------------------------------------------------------
def extract_equations_from_image(image: Image.Image):
    # تحويل الصورة إلى مصفوفة قابلة للمعالجة بواسطة OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. تحويل الصورة إلى تدرج الرمادي (Grayscale)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. تقليل التشويش (Blur) لتحسين القراءة
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 3. زيادة التباين (Thresholding) لجعل النص أسود نقي والخلفية بيضاء
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # تحويل الصورة المعالجة لتقرأها مكتبة Tesseract
    processed_image = Image.fromarray(thresh)

    # إعدادات Tesseract لقراءة المعادلات بشكل أفضل (اعتبار الصورة كتلة واحدة)
# إجبار الـ OCR على قراءة هذه الرموز والأرقام فقط وتجاهل باقي اللغات
    custom_config = r'-c tessedit_char_whitelist=xyzXYZ0123456789+-= --oem 3 --psm 6'    
    # استخراج النص
    raw_text = pytesseract.image_to_string(processed_image, config=custom_config)
    
    # تنظيف وتصحيح بعض أخطاء القراءة الشائعة لبرامج الـ OCR
    raw_text = re.sub(r'[^xyz0-9\+\-\=\.\n]', '', raw_text.lower())    # تقسيم النص لسطور واستخراج المعادلات فقط
    equations = [line.strip() for line in raw_text.splitlines() if line.strip() and "=" in line]
    
    if not equations:
         raise Exception("لم يتم التعرف على معادلات صالحة. تأكد من وضوح الصورة ووجود علامة =")
         
    return equations

# ---------------------------------------------------------
# 3. تحويل نصوص المعادلات إلى مصفوفة موسعة (وتطبيعها لـ x, y, z)
# ---------------------------------------------------------
def parse_equations(equations):
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    target_vars = ['x', 'y', 'z']
    sym_vars = [symbols(v) for v in target_vars]
    
    normalized_eqs = []
    for eq_str in equations:
        eq_norm = eq_str.lower().strip()
        
        # تصحيح الرموز المتلخبطة تلقائياً
        eq_norm = re.sub(r'x_{?1}?|x1|x₁', 'x', eq_norm)
        eq_norm = re.sub(r'x_{?2}?|x2|x₂', 'y', eq_norm)
        eq_norm = re.sub(r'x_{?3}?|x3|x₃', 'z', eq_norm)
        eq_norm = eq_norm.replace("^", "")
        eq_norm = re.sub(r'\s+', '', eq_norm) # مسح المسافات لمنع أخطاء بايثون
        
        lhs, rhs = eq_norm.split("=")
        lhs_expr = parse_expr(lhs, transformations=transformations)
        rhs_expr = parse_expr(rhs, transformations=transformations)
        
        eq = Eq(lhs_expr, rhs_expr)
        normalized_eqs.append(eq)
        
    A, b = linear_eq_to_matrix(normalized_eqs, sym_vars)
    
    rows, cols = A.shape
    matrix = []
    for i in range(rows):
        row = [Fraction(int(A[i, j])) if A[i, j].is_integer else Fraction(str(A[i, j])) for j in range(cols)]
        row.append(Fraction(int(b[i, 0])) if b[i, 0].is_integer else Fraction(str(b[i, 0])))
        matrix.append(row)
        
    return matrix, target_vars

# ---------------------------------------------------------
# 4. دوال التنسيق لعرض LaTeX الأكاديمي
# ---------------------------------------------------------
def format_val(val):
    if isinstance(val, Fraction) and val.denominator != 1:
        if val < 0:
            return f"-\\frac{{{abs(val.numerator)}}}{{{val.denominator}}}"
        return f"\\frac{{{val.numerator}}}{{{val.denominator}}}"
    return str(val)

def matrix_to_latex(mat):
    cols = len(mat[0]) - 1
    latex_str = r"\left[\begin{array}{" + "c" * cols + r"|c}"
    for row in mat:
        row_str = " & ".join(format_val(val) for val in row)
        latex_str += row_str + r" \\ "
    latex_str += r"\end{array}\right]"
    return latex_str

# ---------------------------------------------------------
# 5. محرك الحذف الغاوسي (Step-by-Step Solver)
# ---------------------------------------------------------
def solve_gaussian_elimination(aug_matrix):
    mat = [[val for val in row] for row in aug_matrix]
    steps = []
    num_rows = len(mat)
    num_cols = len(mat[0]) - 1
    
    steps.append({
        "desc": "المصفوفة الموسعة المبدئية $[A|b]$",
        "latex": matrix_to_latex(mat)
    })
    
    lead = 0
    for r in range(num_rows):
        if lead >= num_cols:
            break
            
        i = r
        while i < num_rows and mat[i][lead] == 0:
            i += 1
            
        if i == num_rows:
            lead += 1
            continue
            
        if i != r:
            mat[i], mat[r] = mat[r], mat[i]
            steps.append({
                "desc": f"تبديل الصف {r+1} مع الصف {i+1}: $(R_{{{r+1}}} \\leftrightarrow R_{{{i+1}}})$",
                "latex": matrix_to_latex(mat)
            })
            
        pivot = mat[r][lead]
        if pivot != 1 and pivot != 0:
            mat[r] = [val / pivot for val in mat[r]]
            steps.append({
                "desc": f"قسمة الصف {r+1} على {format_val(pivot)} لإنشاء (Leading 1): $(\\frac{{1}}{{{format_val(pivot)}}} R_{{{r+1}}} \\to R_{{{r+1}}})$",
                "latex": matrix_to_latex(mat)
            })
            
        for j in range(r + 1, num_rows):
            factor = mat[j][lead]
            if factor != 0:
                mat[j] = [mat[j][c] - factor * mat[r][c] for c in range(len(mat[0]))]
                steps.append({
                    "desc": f"تصفير العنصر في الصف {j+1}: $(R_{{{j+1}}} - ({format_val(factor)})R_{{{r+1}}} \\to R_{{{j+1}}})$",
                    "latex": matrix_to_latex(mat)
                })
                
        lead += 1

    steps.append({
        "desc": "تم الوصول إلى صيغة (Row Echelon Form). نبدأ التعويض العكسي.",
        "latex": matrix_to_latex(mat)
    })
    
    # ---------------- التعويض العكسي ----------------
    solution = [0] * num_cols
    for i in range(num_rows - 1, -1, -1):
        all_zeros = all(mat[i][c] == 0 for c in range(num_cols))
        if all_zeros and mat[i][-1] != 0:
            return steps, "لا يوجد حل (النظام متناقض Inconsistent)", None
            
        pivot_col = -1
        for c in range(num_cols):
            if mat[i][c] != 0:
                pivot_col = c
                break
                
        if pivot_col != -1:
            val = mat[i][-1]
            for c in range(pivot_col + 1, num_cols):
                val -= mat[i][c] * solution[c]
            solution[pivot_col] = val / mat[i][pivot_col]
            
    return steps, "تم إيجاد حل فريد بنجاح!", solution

# ---------------------------------------------------------
# 6. واجهة المستخدم والتنفيذ (Streamlit Front-End)
# ---------------------------------------------------------
def display_solution(equations):
    st.success("المعادلات التي سيتم حلها:")
    for eq in equations:
        st.code(eq, language="text")
        
    aug_mat, var_names = parse_equations(equations)
    steps, status, sol = solve_gaussian_elimination(aug_mat)
    
    st.markdown("---")
    st.subheader("خطوات الحل (Step-by-Step):")
    for step_idx, step in enumerate(steps, 1):
        st.markdown(f"**الخطوة {step_idx}:** {step['desc']}")
        st.latex(step["latex"])
        
    st.markdown("---")
    st.subheader("النتيجة النهائية:")
    if sol is not None:
        st.success(status)
        for var, val in zip(var_names, sol):
            st.latex(f"{var.upper()} = {format_val(val)}") # عرض X, Y, Z كابيتال لجمال الشكل
    else:
        st.error(status)


# إنشاء تبويبات (Tabs) لاختيار طريقة الإدخال
tab1, tab2 = st.tabs(["📷 رفع صورة (Image Upload)", "⌨️ كتابة يدوية (Text Input)"])

# ----------------- التبويب الأول: رفع الصورة -----------------
with tab1:
    uploaded_file = st.file_uploader("ارفع صورة المعادلات (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", width=400)
        
        with st.spinner("جاري تحليل الصورة واستخراج المعادلات..."):
            try:
                # تم إزالة api_key من الاستدعاء
                equations = extract_equations_from_image(image)
                display_solution(equations)
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحليل الصورة: {str(e)}")
                st.info("نصيحة: تأكد أن الصورة واضحة وأن المعادلات خطية (مثال: x+y+z=1).")

# ----------------- التبويب الثاني: الإدخال اليدوي -----------------
with tab2:
    st.write("اكتب المعادلات الخطية هنا (معادلة واحدة في كل سطر):")
    st.info("مثال:\nx + y + z = 1\nx - y = 5\nx + z = 4")
    
    manual_input = st.text_area("أدخل المعادلات:", height=150)
    solve_button = st.button("حل المعادلات (Solve)")
    
    if solve_button and manual_input.strip():
        manual_equations = [line.strip() for line in manual_input.splitlines() if "=" in line]
        
        if not manual_equations:
            st.error("لم يتم العثور على معادلات صالحة. تأكد من استخدام علامة '='.")
        else:
            with st.spinner("جاري حل المعادلات..."):
                try:
                    display_solution(manual_equations)
                except Exception as e:
                    st.error(f"حدث خطأ رياضي في المعادلات المدخلة: {str(e)}")
                    st.info("تأكد من كتابة المعادلات بصيغة صحيحة.")