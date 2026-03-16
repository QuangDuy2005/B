import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Dự đoán Cảnh báo học vụ", layout="wide")

# =========================
# CSS GIAO DIỆN
# =========================

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}

.big-font {
    font-size:50px !important;
    font-weight:bold;
}

.prediction {
    font-size:32px;
    color:#ff4b4b;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

try:
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
except:
    st.error("Chưa có model.pkl. Hãy chạy train_model.py trước")
    st.stop()

# =========================
# SIDEBAR INPUT
# =========================

st.sidebar.title("Thông tin sinh viên")

age = st.sidebar.slider("Tuổi",18,30,20)

count_f = st.sidebar.slider("Số môn trượt",0,10,0)

debt = st.sidebar.slider(
    "Nợ học phí",
    0,
    50000000,
    0,
    step=1000000
)

training_score = st.sidebar.slider("Điểm rèn luyện",0,100,85)

club = st.sidebar.selectbox(
    "Tham gia câu lạc bộ",
    ["Yes","No"]
)

english = st.sidebar.selectbox(
    "Trình độ tiếng Anh",
    ["A1","A2","B1","B2","IELTS 6.0+"]
)

# =========================
# PREDICTION
# =========================

input_df = pd.DataFrame({

    "Count_F":[float(count_f)],
    "Tuition_Debt":[float(debt)],
    "Training_Score_Mixed":[float(training_score)],
    "Age":[int(age)],
    "Club_Member":[str(club)],
    "English_Level":[str(english)]

})

prediction = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]

mapping = {
    0:"Bình thường",
    1:"Cảnh báo học vụ",
    2:"Nguy cơ thôi học"
}

# =========================
# MAIN UI
# =========================

st.markdown("<p class='big-font'>Kết quả dự đoán</p>", unsafe_allow_html=True)

st.markdown(
    f"<p class='prediction'>{mapping[prediction]}</p>",
    unsafe_allow_html=True
)

st.write("### Xác suất")

result = {}

for i, cls in enumerate(model.classes_):

    if cls == 0:
        result["Bình thường"] = round(probs[i],3)

    elif cls == 1:
        result["Cảnh báo học vụ"] = round(probs[i],3)

    elif cls == 2:
        result["Nguy cơ thôi học"] = round(probs[i],3)

st.write(result)