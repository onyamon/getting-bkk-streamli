import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- 0. Configuration and State Initialization ---
st.set_page_config(
    page_title="เครื่องมือประเมินราคาอสังหาฯ AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Thai:wght@300;400;500;700&display=swap');

* {
    font-family: 'Inter', 'Noto Sans Thai', sans-serif !important;
}
.stApp {
    background-color: #FAF9F6;
    color: #333333;
}
div.stButton > button {
    background-color: #A84D3A;
    color: #FFFFFF !important;
    font-weight: 600;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 1rem;
    box-shadow: 0 4px 10px rgba(168,77,58,0.25);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #8c3f2f;
    transform: translateY(-2px);
}
.price-card-container {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 35px;
    border: 1px solid #E4E2DD;
    box-shadow: 0 12px 25px rgba(168,77,58,0.08);
    text-align: center;
}
.price-thb {
    font-size: 64px !important;
    color: #A84D3A !important;
    font-weight: 800 !important;
}
.price-million {
    font-size: 24px;
    color: #666666;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State ---
if 'best_model_state' not in st.session_state:
    st.session_state.best_model_state = {
        'model_type': 'None',
        'r2_score': -float('inf'),
        'mae': float('inf'),
        'model': None,
        'features': [],
        'target_col': 'Price (THB)',
        'params': {},
        'is_trained': False
    }

# --- 1. Load Data ---
FILE_NAME = "Bangkok Housing Condo Apartment Prices.csv"
RANDOM_SEED = 42

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"ไม่พบไฟล์ข้อมูล '{path}' หรือเกิดข้อผิดพลาดในการโหลด: {e}")
        return pd.DataFrame()

data = load_data(FILE_NAME)
if data.empty:
    st.stop()

target_default = 'Price (THB)' if 'Price (THB)' in data.columns else data.columns[-1]
UI_INPUT_FEATURES = ['Property Type', 'Location', 'Area (sq. ft.)', 'Bedrooms', 'Bathrooms']
feature_cols_all = [col for col in data.columns if col != target_default]

if not all(f in feature_cols_all for f in UI_INPUT_FEATURES):
    missing = [f for f in UI_INPUT_FEATURES if f not in feature_cols_all]
    st.error(f"ข้อมูลขาดคอลัมน์ที่จำเป็น: {', '.join(missing)}")
    st.stop()

# --- 2. Training Function ---
def train_and_evaluate(df, target_col, feature_cols, model_type, params, test_size):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    le_map = {}
    X_train_processed = X_train.copy()
    for col in categorical_features:
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
        le_map[col] = le

    scaler = StandardScaler()
    X_train_processed = pd.DataFrame(scaler.fit_transform(X_train_processed), columns=feature_cols)

    if model_type == "Decision Tree":
        model = DecisionTreeRegressor(
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split'),
            random_state=RANDOM_SEED
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layers'),
            max_iter=params.get('max_iter'),
            random_state=RANDOM_SEED,
            early_stopping=True
        )

    model.fit(X_train_processed, y_train)
    X_test_processed = X_test.copy()
    for col in categorical_features:
        mapping = {label: index for index, label in enumerate(le_map[col].classes_)}
        X_test_processed[col] = X_test_processed[col].astype(str).apply(lambda x: mapping.get(x, -1))
    X_test_processed = pd.DataFrame(scaler.transform(X_test_processed), columns=feature_cols)
    y_pred = model.predict(X_test_processed)

    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    default_values = {}
    for col in feature_cols:
        if col in categorical_features:
            default_values[col] = X_train[col].mode()[0]
        else:
            default_values[col] = X_train[col].mean()

    model_bundle = {
        'model': model,
        'scaler': scaler,
        'label_encoders': le_map,
        'features': feature_cols,
        'default_values': default_values
    }
    return r2, mae, model_bundle

# --- Sidebar ---
st.sidebar.header("⚙️ ตั้งค่าและฝึกโมเดล AI")
model_type = st.sidebar.radio("เลือกเทคนิค AI:", ("Decision Tree", "Neural Network"))
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)

params = {}
if model_type == "Decision Tree":
    params['max_depth'] = st.sidebar.slider("Max Depth:", 1, 30, 8)
    params['min_samples_split'] = st.sidebar.slider("Min Samples Split:", 2, 20, 5)
else:
    hidden_layer_str = st.sidebar.text_input("Hidden Layers (e.g., 100, 50):", "100, 50")
    try:
        params['hidden_layers'] = tuple(map(int, hidden_layer_str.split(',')))
    except:
        params['hidden_layers'] = (100, 50)
    params['max_iter'] = st.sidebar.slider("Max Iterations:", 200, 1000, 500)

if st.sidebar.button("🚀 ฝึกโมเดลและเริ่มใช้งาน"):
    with st.spinner("กำลังฝึกโมเดล..."):
        r2, mae, model_bundle = train_and_evaluate(
            data, target_default, feature_cols_all, model_type, params, test_size/100.0
        )
        st.session_state.best_model_state = {
            'model_type': model_type,
            'r2_score': r2,
            'mae': mae,
            'model': model_bundle,
            'features': feature_cols_all,
            'is_trained': True
        }
        st.sidebar.success("✅ ฝึกโมเดลสำเร็จ!")
        st.balloons()

# --- Sidebar: Model Status ---
best_state = st.session_state.best_model_state
st.sidebar.markdown("### 🌟 สถานะโมเดลปัจจุบัน")

if best_state['is_trained']:
    st.sidebar.markdown(f"""
    🏅 **{best_state['model_type']} Performance**  
    <span style='font-size:32px; color:#A84D3A; font-weight:700;'>{best_state['r2_score']*100:.2f}%</span>  
    <br>
    ความคลาดเคลื่อนเฉลี่ย (MAE):  
    <span style='font-size:20px; color:#A84D3A; font-weight:600;'>±{best_state['mae']:,.0f} THB</span>
    """, unsafe_allow_html=True)
else:
    st.sidebar.info("ยังไม่มีโมเดลที่ฝึกใช้งาน กรุณาฝึกโมเดลก่อน 🚀")

# --- Prediction UI ---
if not best_state['is_trained']:
    st.info("โปรดกด '🚀 ฝึกโมเดลและเริ่มใช้งาน' ก่อน")
    st.stop()

st.markdown("## 💰 Bangkok Housing Condo Apartment Prices")
st.markdown("---")

col_input, col_output = st.columns([1, 1.2])

with col_input:
    st.subheader("🏡 1. กรอกข้อมูล")
    input_vals = {}

    property_options = ["ยังไม่มีข้อมูลที่กรอก"] + sorted(data['Property Type'].unique().tolist())
    input_vals['Property Type'] = st.selectbox("ประเภทอสังหาริมทรัพย์:", property_options, index=0)

    all_locations = ["ยังไม่มีข้อมูลที่กรอก"] + sorted(data['Location'].unique().tolist())
    input_vals['Location'] = st.selectbox("ทำเล (Location):", all_locations, index=0)

    col_area, col_bed, col_bath = st.columns(3)
    with col_area:
        input_vals['Area (sq. ft.)'] = st.number_input("พื้นที่ (ตร.ฟุต):", min_value=0.0, max_value=10000.0, value=0.0, step=10.0, format="%.0f")
    with col_bed:
        input_vals['Bedrooms'] = st.number_input("จำนวนห้องนอน:", min_value=0, max_value=20, value=0, step=1)
    with col_bath:
        input_vals['Bathrooms'] = st.number_input("จำนวนห้องน้ำ:", min_value=0, max_value=20, value=0, step=1)

    if st.button("✨ ประเมินราคา", use_container_width=True):
        try:
            model_bundle = best_state['model']
            best_features = model_bundle['features']
            final_input = model_bundle['default_values'].copy()

            for key, val in input_vals.items():
                if (isinstance(val, str) and val == "ยังไม่มีข้อมูลที่กรอก") or (isinstance(val, (int, float)) and val == 0):
                    continue
                else:
                    final_input[key] = val

            df_input = pd.DataFrame([final_input])[best_features]
            scaler_loaded = model_bundle['scaler']
            le_loaded = model_bundle['label_encoders']

            df_processed = df_input.copy()
            for col, le in le_loaded.items():
                mapping = {label: index for index, label in enumerate(le.classes_)}
                df_processed[col] = df_input[col].astype(str).apply(lambda x: mapping.get(x, -1))
            df_processed[df_processed.columns] = scaler_loaded.transform(df_processed[df_processed.columns])

            final_model = model_bundle['model']
            pred_price = final_model.predict(df_processed[best_features])[0]

            st.session_state.latest_prediction = pred_price
            st.session_state.prediction_success = True

        except Exception as e:
            st.session_state.prediction_success = False
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

with col_output:
    st.subheader("📈 2. ผลการประเมินราคา")
    predicted_price = st.session_state.get('latest_prediction', None)

    if st.session_state.get('prediction_success', False) and predicted_price is not None:
        price_thb = f"฿{predicted_price:,.0f}"
        price_million = f"{predicted_price/1_000_000:,.2f} ล้านบาท"
        st.markdown(f"""
        <div class="price-card-container">
            <p>ราคาประเมิน</p>
            <h2 class="price-thb">{price_thb}</h2>
            <p class="price-million">{price_million}</p>
            <hr>
            <p>โมเดลที่ใช้: {best_state['model_type']} (R²: {best_state['r2_score']:.2%})</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("กรุณากรอกข้อมูลแล้วกด '✨ ประเมินราคา'")
