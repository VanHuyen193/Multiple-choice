import streamlit as st
import demo

# Tạo giao diện Streamlit
st.title("Multiple Choice Question Answering")

# Nhập câu hỏi
question = st.text_input("Enter the question:")

# Nhập các lựa chọn
st.subheader("Enter the answer choices:")
choices = []
for i in range(5):  # Giả sử bạn muốn cho phép tối đa 5 lựa chọn
    choice = st.text_input(f"Choice {i + 1}:", key=f"choice_{i}")
    if choice:
        choices.append(choice.strip())

if st.button("Get Prediction"):
    if question and choices:
        prediction = demo.predict_answer(question, choices)
        st.subheader("Prediction Result:")
        for choice, probability in prediction.items():
            st.write(f"{choice} (Probability: {probability:.4f})")
    else:
        st.warning("Please enter both the question and the answer choices.")
