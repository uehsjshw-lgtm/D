import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 페이지 설정
st.set_page_config(page_title="식빵 품질 학습소", page_icon="🍞")

# 클래스 정의 (알파벳 순서 권장: 변색, 이물, 곰팡이, 정상)
CLASS_NAMES = ['discolor', 'foreign', 'mold', 'normal']

# 모델 로드 함수
@st.cache_resource
def load_initial_model():
    model_path = 'bread_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # 모델 파일이 없을 경우 임시 모델 생성 (과제 제출 시에는 학습된 h5 필수)
        st.error("기본 모델 파일(bread_model.h5)을 찾을 수 없습니다.")
        return None

# 세션 상태에 모델 저장 (실시간 학습 반영을 위해)
if 'current_model' not in st.session_state:
    st.session_state.current_model = load_initial_model()

model = st.session_state.current_model

st.title("🍞 성장형 식빵 품질 판별기")
st.info("외부 DB 없이 서버 내에서 실시간 학습을 진행합니다. (서버 재시작 시 초기화 주의)")

# 이미지 업로드
uploaded_file = st.file_uploader("검사할 식빵 사진을 선택하세요", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    # 1. 이미지 표시 및 전처리
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_container_width=True)
    
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 2. 결과 예측
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100
    
    st.subheader(f"판별 결과: {CLASS_NAMES[pred_idx]} ({confidence:.2f}%)")
    
    # 3. 실시간 재학습 섹션
    st.divider()
    st.write("### 🤖 모델 재학습 (Online Learning)")
    st.write("예측이 틀렸나요? 정답을 알려주면 인공지능이 즉시 학습합니다.")
    
    correct_label = st.selectbox("실제 정답 선택", CLASS_NAMES, index=pred_idx)
    
    if st.button("이 정답으로 모델 강화하기"):
        with st.spinner("방금 올린 사진으로 모델을 교육 중입니다..."):
            # 정답 데이터를 One-hot encoding으로 변환
            target = np.zeros((1, 4))
            target[0, CLASS_NAMES.index(correct_label)] = 1
            
            # 딱 5번만 반복 학습 (서버 부하 방지용)
            model.fit(img_array, target, epochs=5, verbose=0)
            st.session_state.current_model = model
            st.success(f"학습 완료! 이제 이 인공지능은 해당 이미지를 '{correct_label}'로 기억합니다.")

# 4. 모델 저장 및 내보내기 (매우 중요)
st.sidebar.title("🛠️ 모델 관리")
st.sidebar.write("서버가 초기화되면 학습 데이터가 사라집니다. 학습된 모델을 보관하려면 아래 버튼을 누르세요.")

if st.sidebar.button("현재 모델 파일(h5) 생성"):
    model.save("updated_bread_model.h5")
    with open("updated_bread_model.h5", "rb") as f:
        st.sidebar.download_button(
            label="강화된 모델 다운로드",
            data=f,
            file_name="updated_bread_model.h5",
            mime="application/octet-stream"
        )
