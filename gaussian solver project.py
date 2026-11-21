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

# Windows Tesseract path (Linux/Server uses system PATH automatically)
if sys.platform == 'win32':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------------------------------------------------
# 1. Page Configuration & Custom Styling (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Linear Algebra Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.16), rgba(34, 211, 238, 0.08));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 0;
    }

    .eq-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.16);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .eq-equals {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 70px;
        font-size: 1.35rem;
        font-weight: 700;
        color: #22D3EE;
    }

    /* Input Fields */
    div[data-testid="stTextInput"] input {
        background-color: rgba(30, 41, 59, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
        font-size: 0.95rem !important;
        padding: 0.45rem 0.7rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: rgba(30, 41, 59, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 16px !important;
        color: #94A3B8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #22D3EE !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #22D3EE !important;
        height: 2.5px !important;
    }

    /* Primary Solve Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.3) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-top: 8px;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.42) !important;
    }

    /* Step Card */
    .step-card {
        background: rgba(15, 23, 42, 0.55);
        border-left: 3px solid #6366F1;
        border-radius: 10px;
        padding: 11px 16px;
        margin-bottom: 10px;
    }
    .step-title {
        color: #E2E8F0;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🧮 Step-by-Step Linear System Solver</div>
    <p class="hero-sub">Solve systems of linear equations using Gaussian or Gauss-Jordan Elimination with full matrix steps and special case detection.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Extract Equations from Image via Tesseract OCR
# ---------------------------------------------------------
def extract_equations_from_image(image: Image.Image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_image = Image.fromarray(thresh)

    custom_config = r'-c tessedit_char_whitelist=xyzXYZ0123456789+-=._ --oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(processed_image, config=custom_config)

    cleaned_lines = []
    for line in raw_text.splitlines():
        line_clean = re.sub(r'[^a-zA-Z0-9\+\-\=\.\s]', '', line).strip()
        if "=" in line_clean:
            cleaned_lines.append(line_clean)

    if not cleaned_lines:
        raise Exception("No valid equations detected. Make sure the image is clear and contains '=' signs.")

    return cleaned_lines

# ---------------------------------------------------------
# 3. Parse Equation Strings into Augmented Matrix
# ---------------------------------------------------------
def normalize_equation_text(eq_str: str, target_vars: list[str]) -> str:
    eq_norm = eq_str.lower().strip()
    eq_norm = re.sub(r'x_\{?(\d+)\}?', r'x\1', eq_norm)
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    eq_norm = eq_norm.translate(subscripts)

    legacy_map = {'x': 'x1', 'y': 'x2', 'z': 'x3'}
    if all(v.startswith('x') and v[1:].isdigit() for v in target_vars):
        if not re.search(r'\bx\d+\b', eq_norm):
            for k, v in legacy_map.items():
                if v in target_vars:
                    eq_norm = re.sub(rf'(?<![a-zA-Z0-9_]){k}(?![a-zA-Z0-9_])', v, eq_norm)

    eq_norm = eq_norm.replace("^", "**")
    eq_norm = re.sub(r'\s+', '', eq_norm)
    return eq_norm

def parse_equations_from_text(equations: list[str], num_vars: int):
    transformations = standard_transformations + (implicit_multiplication_application,)
    target_vars = [f"x{i+1}" for i in range(num_vars)]
    sym_vars = [symbols(v) for v in target_vars]

    normalized_eqs = []
    for eq_str in equations:
        if eq_str.count("=") != 1:
            raise ValueError(f"Invalid equation format: {eq_str}")
        eq_norm = normalize_equation_text(eq_str, target_vars)
        lhs, rhs = eq_norm.split("=")
        if not lhs or not rhs:
            raise ValueError(f"Both sides of the equation must be non-empty: {eq_str}")

        lhs_expr = parse_expr(lhs, transformations=transformations)
        rhs_expr = parse_expr(rhs, transformations=transformations)
        normalized_eqs.append(Eq(lhs_expr, rhs_expr))

    A, b = linear_eq_to_matrix(normalized_eqs, sym_vars)
    rows, cols = A.shape
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            val = A[i, j]
            row.append(Fraction(int(val)) if val.is_integer else Fraction(str(val)))
        rhs_val = b[i, 0]
        row.append(Fraction(int(rhs_val)) if rhs_val.is_integer else Fraction(str(rhs_val)))
        matrix.append(row)

    return matrix, target_vars

def parse_number_to_fraction(raw_val: str, default_zero=True) -> Fraction:
    cleaned = raw_val.strip()
    if cleaned == "":
        return Fraction(0) if default_zero else None
    return Fraction(cleaned)

# ---------------------------------------------------------
# 4. LaTeX Formatting Helpers
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
# 5. Gaussian & Gauss-Jordan Elimination Engine
# ---------------------------------------------------------
def analyze_system(mat, num_cols, var_names):
    num_rows = len(mat)
    pivot_cols = []

    for r in range(num_rows):
        if all(mat[r][c] == 0 for c in range(num_cols)) and mat[r][-1] != 0:
            return "no_solution", "No Solution (The system is inconsistent).", None

        for c in range(num_cols):
            if mat[r][c] != 0:
                pivot_cols.append((r, c))
                break

    used_cols = [c for _, c in pivot_cols]
    free_cols = [c for c in range(num_cols) if c not in used_cols]

    if free_cols:
        free_names = [var_names[c].upper() for c in free_cols]
        expressions = {}
        for c in free_cols:
            expressions[c] = var_names[c].upper()

        for r, p_col in reversed(pivot_cols):
            rhs = mat[r][-1]
            terms = []
            if rhs != 0:
                terms.append(format_val(rhs))

            for c in range(p_col + 1, num_cols):
                coeff = mat[r][c]
                if coeff != 0:
                    neg_coeff = -coeff
                    var_label = expressions.get(c, var_names[c].upper())
                    if neg_coeff > 0:
                        prefix = " + " if terms else ""
                        coeff_str = "" if neg_coeff == 1 else f"{format_val(neg_coeff)}"
                        terms.append(f"{prefix}{coeff_str}{var_label}")
                    else:
                        abs_c = abs(neg_coeff)
                        coeff_str = "" if abs_c == 1 else f"{format_val(abs_c)}"
                        terms.append(f" - {coeff_str}{var_label}")

            expr_str = "".join(terms).strip() if terms else "0"
            if expr_str.startswith(" + "):
                expr_str = expr_str[3:]
            expressions[p_col] = expr_str

        return (
            "infinite_solutions",
            f"Infinite Solutions — Free variable(s): {', '.join(free_names)}",
            expressions
        )

    if len(pivot_cols) < num_cols:
        return "infinite_solutions", "Infinite Solutions.", {}

    solution = [Fraction(0)] * num_cols
    for r, p_col in reversed(pivot_cols):
        val = mat[r][-1]
        for c in range(p_col + 1, num_cols):
            val -= mat[r][c] * solution[c]
        solution[p_col] = val / mat[r][p_col]

    return "unique", "Unique solution found!", solution

def solve_linear_system(aug_matrix, method="Gauss-Jordan Elimination"):
    mat = [[val for val in row] for row in aug_matrix]
    steps = []
    num_rows = len(mat)
    num_cols = len(mat[0]) - 1

    steps.append({
        "desc": "Initial Augmented Matrix $[A|b]$",
        "latex": matrix_to_latex(mat)
    })

    # Phase 1: Forward Elimination to Row Echelon Form (REF)
    lead = 0
    pivot_positions = []

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
                "desc": f"Swap Row {r+1} with Row {i+1}: $(R_{{{r+1}}} \\leftrightarrow R_{{{i+1}}})$",
                "latex": matrix_to_latex(mat)
            })

        pivot = mat[r][lead]
        if pivot != 1 and pivot != 0:
            mat[r] = [val / pivot for val in mat[r]]
            steps.append({
                "desc": f"Divide Row {r+1} by {format_val(pivot)} to create Leading 1: $(\\frac{{1}}{{{format_val(pivot)}}} R_{{{r+1}}} \\to R_{{{r+1}}})$",
                "latex": matrix_to_latex(mat)
            })

        for j in range(r + 1, num_rows):
            factor = mat[j][lead]
            if factor != 0:
                mat[j] = [mat[j][c] - factor * mat[r][c] for c in range(len(mat[0]))]
                steps.append({
                    "desc": f"Eliminate entry below pivot in Row {j+1}, Column {lead+1}: $(R_{{{j+1}}} - ({format_val(factor)})R_{{{r+1}}} \\to R_{{{j+1}}})$",
                    "latex": matrix_to_latex(mat)
                })

        pivot_positions.append((r, lead))
        lead += 1

    steps.append({
        "desc": "Reached Row Echelon Form (REF).",
        "latex": matrix_to_latex(mat)
    })

    # Phase 2: Backward Elimination for Gauss-Jordan (RREF)
    if method == "Gauss-Jordan Elimination":
        for r, p_col in reversed(pivot_positions):
            if mat[r][p_col] != 0 and mat[r][p_col] != 1:
                piv = mat[r][p_col]
                mat[r] = [val / piv for val in mat[r]]
                steps.append({
                    "desc": f"Normalize pivot in Row {r+1} to 1: $(\\frac{{1}}{{{format_val(piv)}}} R_{{{r+1}}} \\to R_{{{r+1}}})$",
                    "latex": matrix_to_latex(mat)
                })

            for above_r in range(r - 1, -1, -1):
                factor = mat[above_r][p_col]
                if factor != 0:
                    mat[above_r] = [mat[above_r][c] - factor * mat[r][c] for c in range(len(mat[0]))]
                    steps.append({
                        "desc": f"Eliminate entry above pivot in Row {above_r+1}: $(R_{{{above_r+1}}} - ({format_val(factor)})R_{{{r+1}}} \\to R_{{{above_r+1}}})$",
                        "latex": matrix_to_latex(mat)
                    })

        steps.append({
            "desc": "Reached Reduced Row Echelon Form (RREF).",
            "latex": matrix_to_latex(mat)
        })

    status_type, status_msg, sol_data = analyze_system(mat, num_cols, [f"x{i+1}" for i in range(num_cols)])
    return steps, status_type, status_msg, sol_data

# ---------------------------------------------------------
# 6. Render Steps & Final Result
# ---------------------------------------------------------
def render_steps_and_solution(aug_mat, var_names, method):
    steps, status_type, status_msg, sol_data = solve_linear_system(aug_mat, method=method)

    st.markdown("---")
    st.subheader("Step-by-Step Solution")
    for step_idx, step in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step {step_idx}: {step["desc"]}</div></div>',
            unsafe_allow_html=True
        )
        st.latex(step["latex"])

    st.markdown("---")
    st.subheader("Final Result")
    if status_type == "unique":
        st.success(status_msg)
        res_cols = st.columns(min(len(var_names), 4))
        for idx, (var, val) in enumerate(zip(var_names, sol_data)):
            with res_cols[idx % len(res_cols)]:
                st.latex(f"{var.upper()} = {format_val(val)}")
    elif status_type == "infinite_solutions":
        st.warning(status_msg)
        if sol_data:
            res_cols = st.columns(min(len(var_names), 4))
            for idx, var in enumerate(var_names):
                with res_cols[idx % len(res_cols)]:
                    st.latex(f"{var.upper()} = {sol_data.get(idx, var.upper())}")
    else:
        st.error(status_msg)

# ---------------------------------------------------------
# 7. Main UI: Tabs at Top -> Variables & Equations Config
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔢 Coefficients Grid", "📷 Image Upload"])

with tab1:
    with st.container(border=True):
        cfg_col1, cfg_col2, cfg_col3 = st.columns([1, 1, 1.4])

        with cfg_col1:
            num_vars = st.number_input(
                "Number of Variables",
                min_value=1,
                max_value=15,
                value=3,
                step=1
            )

        with cfg_col2:
            default_eqs = min(3, int(num_vars))
            num_eqs = st.number_input(
                "Number of Equations",
                min_value=1,
                max_value=int(num_vars),
                value=default_eqs,
                step=1
            )

        with cfg_col3:
            method = st.selectbox(
                "Elimination Method",
                ["Gauss-Jordan Elimination", "Gaussian Elimination"]
            )

    var_labels = [f"X{i+1}" for i in range(int(num_vars))]
    var_names = [f"x{i+1}" for i in range(int(num_vars))]

    st.caption("Enter the coefficient for each variable and the right-hand side (RHS) constant after '='. Supports integers and fractions like 1/2 (defaults to 0).")

    aug_matrix_input = []
    for eq_idx in range(int(num_eqs)):
        with st.container(border=True):
            st.markdown(f'<span class="eq-badge">Equation {eq_idx + 1}</span>', unsafe_allow_html=True)
            cols = st.columns(int(num_vars) + 2)
            row_vals = []

            for v_idx in range(int(num_vars)):
                with cols[v_idx]:
                    val_str = st.text_input(
                        label=f"{var_labels[v_idx]} coeff",
                        value="0",
                        key=f"coeff_{eq_idx}_{v_idx}"
                    )
                    row_vals.append(val_str)

            with cols[int(num_vars)]:
                st.markdown('<div class="eq-equals">=</div>', unsafe_allow_html=True)

            with cols[int(num_vars) + 1]:
                rhs_str = st.text_input(
                    label="RHS",
                    value="0",
                    key=f"rhs_{eq_idx}"
                )

            aug_matrix_input.append((row_vals, rhs_str))

    solve_btn = st.button("Solve System", type="primary")

    if solve_btn:
        try:
            parsed_matrix = []
            preview_eqs = []

            for eq_idx, (coeffs_str, rhs_str) in enumerate(aug_matrix_input):
                parsed_row = []
                eq_parts = []

                for v_idx, c_str in enumerate(coeffs_str):
                    frac_c = parse_number_to_fraction(c_str, default_zero=True)
                    parsed_row.append(frac_c)
                    if frac_c != 0:
                        sign = "+" if (frac_c > 0 and eq_parts) else ("-" if frac_c < 0 else "")
                        abs_c = abs(frac_c)
                        c_disp = "" if abs_c == 1 else str(abs_c)
                        if frac_c < 0 and not eq_parts:
                            eq_parts.append(f"-{c_disp}{var_labels[v_idx]}")
                        else:
                            eq_parts.append(f"{sign} {c_disp}{var_labels[v_idx]}".strip())

                frac_rhs = parse_number_to_fraction(rhs_str, default_zero=True)
                parsed_row.append(frac_rhs)
                parsed_matrix.append(parsed_row)

                lhs_text = " ".join(eq_parts) if eq_parts else "0"
                preview_eqs.append(f"{lhs_text} = {frac_rhs}")

            st.success("Constructed Linear System:")
            for eq_txt in preview_eqs:
                st.code(eq_txt, language="text")

            render_steps_and_solution(parsed_matrix, var_names, method)

        except Exception as e:
            st.error(f"Please enter valid numbers or fractions in all fields: {str(e)}")

with tab2:
    with st.container(border=True):
        img_col1, img_col2 = st.columns([1.4, 1])
        with img_col1:
            img_method = st.selectbox(
                "Elimination Method",
                ["Gauss-Jordan Elimination", "Gaussian Elimination"],
                key="img_method"
            )
        with img_col2:
            img_num_vars = st.number_input(
                "Number of Variables",
                min_value=1,
                max_value=15,
                value=3,
                step=1,
                key="img_num_vars"
            )

    uploaded_file = st.file_uploader("Upload Equations Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=380)
        with st.spinner("Extracting equations from image..."):
            try:
                equations = extract_equations_from_image(image)
                st.success("Extracted Equations:")
                for eq in equations:
                    st.code(eq, language="text")
                aug_mat, parsed_vars = parse_equations_from_text(equations, int(img_num_vars))
                render_steps_and_solution(aug_mat, parsed_vars, img_method)
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")