import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import datetime

# --- 0. Configuration and State Initialization ---
st.set_page_config(
    page_title="เครื่องมือประเมินราคาอสังหาริมทรัพย์(Decision Tree)", 
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
/* Style adjustments for cleaner output based on user request */
.r2-info {
    font-size: 14px; 
    color: #999; 
    margin-bottom: 5px;
}
.mae-range-info {
    font-weight: 600; 
    font-size: 18px;
    margin: 10px 0;
}
.range-info {
    color: #666; 
    font-size: 14px;
}
.similar-count {
    color: #999; 
    font-size: 12px;
    margin-top: 5px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State ---
if 'best_model_state' not in st.session_state:
    st.session_state.best_model_state = {
        'model_type': 'Decision Tree',
        'r2_score': None,
        'mae': None, 
        'model': None,
        'features': [],
        'target_col': 'Price (THB)',
        'is_trained': False,
        'original_data': pd.DataFrame() # Store original data for similar search
    }
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {
        'latest_prediction': None,
        'current_mae': None,
        'price_bounds': None,
        'similar_count': 0, # ยังคงเก็บค่าไว้ แต่ไม่ได้แสดงผล
        'prediction_success': False
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

# --- 2. Training Function (Decision Tree + One-Hot Encoding) ---
def train_and_evaluate(df, target_col, feature_cols, params, test_size):
    # One-Hot Encoding (OHE) on the full dataset before splitting
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Ensure only relevant columns are OHE
    df_encoded = pd.get_dummies(df, columns=[c for c in categorical_features if c in feature_cols], drop_first=True)
    
    # Identify all features used after OHE
    updated_feature_cols = [col for col in df_encoded.columns if col != target_col and col in feature_cols or any(f in col for f in categorical_features)]

    X = df_encoded[updated_feature_cols]
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=updated_feature_cols, index=X_train.index)
    
    model = DecisionTreeRegressor(
        max_depth=params.get('max_depth'),
        min_samples_split=params.get('min_samples_split'),
        random_state=RANDOM_SEED
    )

    model.fit(X_train_scaled, y_train)

    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=updated_feature_cols, index=X_test.index)
    
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred) 

    default_values_input = {}
    for col in feature_cols:
        if col in df.select_dtypes(include=['object', 'category']).columns:
            default_values_input[col] = df[col].mode()[0]
        else:
            default_values_input[col] = df[col].mean()

    model_bundle = {
        'model': model,
        'scaler': scaler,
        'ohe_features': updated_feature_cols,
        'original_features': feature_cols,
        'default_values': default_values_input
    }
    return r2, mae_test, model_bundle

# --- ฟังก์ชันคำนวณความคลาดเคลื่อนจากข้อมูลใกล้เคียง (นำกลับมาใช้) ---
def calculate_similar_mae(input_data, original_data, predicted_price, target_col='Price (THB)', top_n=20):
    """
    คำนวณความคลาดเคลื่อนเฉลี่ยและช่วงราคาจากข้อมูลที่คล้ายกันในชุดข้อมูลเดิม
    """
    try:
        # 1. การกรองเบื้องต้น: กรองตามประเภทอสังหาฯ และทำเลที่ตั้ง
        similar_mask = pd.Series([True] * len(original_data))
        
        if 'Property Type' in input_data and input_data['Property Type'] != "ยังไม่มีข้อมูลที่กรอก":
            similar_mask &= (original_data['Property Type'] == input_data['Property Type'])
        
        if 'Location' in input_data and input_data['Location'] != "ยังไม่มีข้อมูลที่กรอก":
            similar_mask &= (original_data['Location'] == input_data['Location'])
        
        similar_data = original_data[similar_mask].copy()
        
        # 2. กรณีที่ข้อมูลที่กรองมีน้อยเกินไป
        if len(similar_data) < 5:
            # ใช้ข้อมูลทั้งหมดและให้ความสำคัญกับค่าตัวเลขมากขึ้น
            similar_data = original_data.copy()
            
        # 3. คำนวณระยะทางความแตกต่างจากค่าตัวเลข
        # ใช้ค่า 0 ถ้าคอลัมน์ไม่มีใน input data (ซึ่งไม่ควรเกิดขึ้นถ้าใช้ UI_INPUT_FEATURES)
        area = input_data.get('Area (sq. ft.)', similar_data['Area (sq. ft.)'].mean())
        bed = input_data.get('Bedrooms', similar_data['Bedrooms'].mean())
        bath = input_data.get('Bathrooms', similar_data['Bathrooms'].mean())

        similar_data['area_diff'] = abs(similar_data['Area (sq. ft.)'] - area)
        similar_data['bed_diff'] = abs(similar_data['Bedrooms'] - bed)
        similar_data['bath_diff'] = abs(similar_data['Bathrooms'] - bath)
        
        # Normalize numerical differences and calculate similarity score
        max_area = similar_data['Area (sq. ft.)'].max() if similar_data['Area (sq. ft.)'].max() > 0 else 1
        
        similar_data['similarity_score'] = (
            similar_data['area_diff'] / max_area + 
            similar_data['bed_diff'] / (similar_data['Bedrooms'].max() + 1) +
            similar_data['bath_diff'] / (similar_data['Bathrooms'].max() + 1)
        )
        
        # 4. เรียงตามความคล้ายคลึงและเลือก Top N
        # ใช้ nsmalles (เลือกค่าที่น้อยที่สุด = คล้ายที่สุด)
        similar_data = similar_data.nsmallest(top_n, 'similarity_score')
        
        if similar_data.empty:
              return {
                 'mae': predicted_price * 0.1,
                 'lower_bound': max(0, predicted_price * 0.9),
                 'upper_bound': predicted_price * 1.1,
                 'similar_count': 0,
             }

        # 5. คำนวณความคลาดเคลื่อนและช่วงราคา
        actual_prices = similar_data[target_col].values
        
        # MAE: Mean Absolute Error จากราคาที่คาดการณ์กับราคาจริงของ Top N
        mae = np.mean(np.abs(actual_prices - predicted_price))
        
        # ช่วงราคา: ใช้ MAE ที่คำนวณได้เป็นความคลาดเคลื่อนที่แสดงผล
        lower_bound = predicted_price - mae
        upper_bound = predicted_price + mae
        
        return {
            'mae': mae,
            'lower_bound': max(0, lower_bound),
            'upper_bound': upper_bound,
            'similar_count': len(similar_data),
        }
    except Exception as e:
        # Fallback in case of error
        print(f"Error in calculate_similar_mae: {e}")
        return {
            'mae': predicted_price * 0.1,
            'lower_bound': max(0, predicted_price * 0.9),
            'upper_bound': predicted_price * 1.1,
            'similar_count': 0,
        }

# --- Sidebar ---
model_type = "Decision Tree" # Locket to Decision Tree
st.sidebar.header("⚙️ ตั้งค่าและฝึกโมเดล Decision Tree")
st.sidebar.info(f"เทคนิคที่ใช้: **{model_type}**")

test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)

params = {}
params['max_depth'] = st.sidebar.slider(
    "1. Max Depth :", 
    1, 
    30, 
    8,
    help="ควบคุมความซับซ้อนของโมเดล ค่าที่สูงอาจทำให้เกิด Overfitting"
)
params['min_samples_split'] = st.sidebar.slider(
    "2. Min Samples Split :", 
    2, 
    20, 
    5,
    help="ค่าที่สูงขึ้นจะช่วยให้โมเดลเรียบง่ายและเสถียรขึ้น"
)


if st.sidebar.button("🚀 ฝึกโมเดลและเริ่มใช้งาน"):
    # Reset prediction results
    st.session_state.prediction_results = {
        'latest_prediction': None,
        'current_mae': None,
        'price_bounds': None,
        'similar_count': 0,
        'prediction_success': False
    }
    
    with st.spinner("กำลังฝึกโมเดล Decision Tree..."):
        r2, mae_test, model_bundle = train_and_evaluate(
            data, target_default, feature_cols_all, params, test_size
        )
        # Store R2 and MAE from training
        st.session_state.best_model_state = {
            'model_type': model_type,
            'r2_score': r2,
            'mae': mae_test, # MAE from Test Set
            'model': model_bundle,
            'features': feature_cols_all,
            'is_trained': True,
            'original_data': data # Store original data here
        }
        st.sidebar.success("✅ ฝึกโมเดลสำเร็จ!")
        st.balloons()
        st.rerun()

# --- Sidebar: Model Status ---
best_state = st.session_state.best_model_state
st.sidebar.markdown("### 🌟 สถานะโมเดลปัจจุบัน")

if best_state['is_trained'] and best_state['r2_score'] is not None:
    st.sidebar.markdown(f"""
    🏅 **{best_state['model_type']} Performance** <span style='font-size:32px; color:#A84D3A; font-weight:700;'>{best_state['r2_score']*100:.2f}%</span>  
    <p style='font-size:14px; color:#666666;'>R² Score (Test Set)</p>
    """, unsafe_allow_html=True)
    
    # แสดง MAE จากการประเมินราคาล่าสุด
    current_mae = st.session_state.prediction_results.get('current_mae')
    if st.session_state.prediction_results.get('prediction_success', False) and current_mae is not None:
        st.sidebar.markdown(f"""
        
        <span style='font-size:16px; color:#333; font-weight:600;'>ความคลาดเคลื่อนการประเมิน:</span>  
        <span style='font-size:20px; color:#A84D3A; font-weight:600;'>±{current_mae:,.0f} THB</span>
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

    # Set input ranges based on data
    max_area_input = data['Area (sq. ft.)'].max() * 2 
    max_bedrooms_input = int(data['Bedrooms'].max() + 5) if 'Bedrooms' in data.columns else 10
    max_bathrooms_input = int(data['Bathrooms'].max() + 5) if 'Bathrooms' in data.columns else 10


    property_options = ["ยังไม่มีข้อมูลที่กรอก"] + sorted(data['Property Type'].unique().tolist())
    input_vals['Property Type'] = st.selectbox("ประเภทอสังหาริมทรัพย์:", property_options, index=0)

    all_locations = ["ยังไม่มีข้อมูลที่กรอก"] + sorted(data['Location'].unique().tolist())
    input_vals['Location'] = st.selectbox("ทำเล (Location):", all_locations, index=0)

    col_area, col_bed, col_bath = st.columns(3)
    with col_area:
        input_vals['Area (sq. ft.)'] = st.number_input(
            "พื้นที่ (ตร.ฟุต):", 
            min_value=0.0, 
            max_value=float(max_area_input), 
            value=0.0, 
            step=10.0, 
            format="%.0f"
        )
    with col_bed:
        input_vals['Bedrooms'] = st.number_input(
            "จำนวนห้องนอน:", 
            min_value=0, 
            max_value=max_bedrooms_input, 
            value=0, 
            step=1
        )
    with col_bath:
        input_vals['Bathrooms'] = st.number_input(
            "จำนวนห้องน้ำ:", 
            min_value=0, 
            max_value=max_bathrooms_input, 
            value=0, 
            step=1
        )

    if st.button("✨ ประเมินราคา", use_container_width=True):
        st.session_state.prediction_results['prediction_success'] = False # Reset flag before processing

        # --- START NEW: Input Validation Check ---
        is_valid = True
        missing_fields = []

        # Check Categorical Fields
        if input_vals['Property Type'] == "ยังไม่มีข้อมูลที่กรอก":
            missing_fields.append("ประเภทอสังหาริมทรัพย์")
            is_valid = False
        
        if input_vals['Location'] == "ยังไม่มีข้อมูลที่กรอก":
            missing_fields.append("ทำเล (Location)")
            is_valid = False
            
        # Check Numerical Fields (must be > 0)
        # Area (sq. ft.) > 0
        if input_vals['Area (sq. ft.)'] <= 0:
            missing_fields.append("พื้นที่ (ตร.ฟุต)")
            is_valid = False
            
        # Bedrooms > 0
        if input_vals['Bedrooms'] <= 0:
            missing_fields.append("จำนวนห้องนอน")
            is_valid = False
            
        # Bathrooms > 0
        if input_vals['Bathrooms'] <= 0:
            missing_fields.append("จำนวนห้องน้ำ")
            is_valid = False

        if not is_valid:
            st.error(f"⚠️ กรุณากรอกข้อมูลให้ครบถ้วนก่อนประเมินราคา: **{', '.join(missing_fields)}**")
        # --- END NEW: Input Validation Check ---
        else:
            # Prediction Logic is now inside the 'else' block
            try:
                model_bundle = best_state['model']
                original_data = best_state['original_data']
                original_features = model_bundle['original_features']
                ohe_features = model_bundle['ohe_features']
                final_input = model_bundle['default_values'].copy()
                
                # --- 1. Merge user input with default values (using original feature names) ---
                user_input_data = {}
                for key, val in input_vals.items():
                    is_empty = (isinstance(val, str) and val == "ยังไม่มีข้อมูลที่กรอก") or \
                               (isinstance(val, (int, float)) and val <= 0 and key != 'Area (sq. ft.)') 
                    
                    if not is_empty:
                        final_input[key] = val
                        user_input_data[key] = val
                    else:
                        # ใช้ค่า Default จาก Training ถ้า Input เป็นค่าว่าง (สำหรับคอลัมน์ที่ไม่ใช่ UI_INPUT_FEATURES หรือในกรณีฉุกเฉิน)
                        final_input[key] = model_bundle['default_values'][key]


                # --- 2. Create DataFrame and apply One-Hot Encoding (OHE) ---
                df_input = pd.DataFrame([final_input])[original_features]
                df_processed = pd.get_dummies(df_input, columns=['Property Type', 'Location'], drop_first=True)
                
                # --- 3. Align Columns with the trained model's OHE features ---
                df_aligned = pd.DataFrame(0, index=[0], columns=ohe_features)
                for col in df_processed.columns:
                    if col in df_aligned.columns:
                        df_aligned[col] = df_processed[col].values[0]

                # --- 4. Apply Scaling ---
                scaler_loaded = model_bundle['scaler']
                df_scaled = pd.DataFrame(scaler_loaded.transform(df_aligned), columns=ohe_features)

                # --- 5. Predict ---
                final_model = model_bundle['model']
                pred_price = final_model.predict(df_scaled)[0]

                if pred_price < 0:
                    pred_price = 100000 
                    
                # --- 6. Calculate Similarity MAE and Bounds ---
                similarity_results = calculate_similar_mae(
                    user_input_data, 
                    original_data, 
                    pred_price,
                    target_default,
                    top_n=20 # ค่านี้ถูกเก็บไว้ในฟังก์ชัน แต่ไม่ได้แสดงผล
                )

                st.session_state.prediction_results['latest_prediction'] = pred_price
                st.session_state.prediction_results['current_mae'] = similarity_results['mae']
                st.session_state.prediction_results['price_bounds'] = (similarity_results['lower_bound'], similarity_results['upper_bound'])
                st.session_state.prediction_results['similar_count'] = similarity_results['similar_count']
                st.session_state.prediction_results['prediction_success'] = True
                st.rerun()

            except Exception as e:
                st.session_state.prediction_results['prediction_success'] = False
                st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

with col_output:
    st.subheader("📈 2. ผลการประเมินราคา")
    results = st.session_state.prediction_results
    predicted_price = results.get('latest_prediction', None)

    if results.get('prediction_success', False) and predicted_price is not None:
        
        price_thb = f"฿{predicted_price:,.0f}"
        price_million = f"{predicted_price/1_000_000:,.2f} ล้านบาท"
        current_mae = results.get('current_mae', 0)
        lower_bound, upper_bound = results.get('price_bounds', (predicted_price, predicted_price))
        r2_value = best_state['r2_score']
        
        # Display the result in the requested format (ไม่มีส่วนของ similar-count)
        st.markdown(f"""
        <div class="price-card-container">
            <p>ราคาประเมิน</p>
            <h2 class="price-thb">{price_thb}</h2>
            <p class="price-million">{price_million}</p>
            <hr>
            <p class="r2-info">โมเดลที่ใช้: {best_state['model_type']} (R²: {r2_value:.2%})</p>
            <p class="mae-range-info" style="color: #A84D3A;">ความคลาดเคลื่อนเฉลี่ย: ±{current_mae:,.0f} THB</p>
            <p class="range-info">ช่วงราคาที่เป็นไปได้: ฿{lower_bound:,.0f} - ฿{upper_bound:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("กรุณากรอกข้อมูลแล้วกด '✨ ประเมินราคา'")
