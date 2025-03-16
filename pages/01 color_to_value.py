import streamlit as st
import cv2
import pandas as pd
# from PIL import Image
# import numpy as np

def main():
    st.title("Image Upload and Display")

    # 文件上傳
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 將上傳的文件保存到本地
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 使用 OpenCV 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            st.error("Error: Unable to read the image file.")
            return

        # 將圖像從 BGR 轉換為 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用 PIL 將圖像轉換為可顯示的格式
        # image_pil = Image.fromarray(image_rgb)

        # 顯示圖像
        st.image(image_path, caption="Uploaded Image", use_container_width=True)
        st.write("Image uploaded and displayed successfully.")

        st.write("Click on the image to get RGB values and specify corresponding value")
        # clicked = st.image(image_path, use_container_width=True)
        # if clicked:

        x = st.number_input("Enter x coordinate", min_value=0, max_value=image.shape[1]-1)
        y = st.number_input("Enter y coordinate", min_value=0, max_value=image.shape[0]-1)
        st.markdown("Max x: " + str(image.shape[1]-1) + " Max y: " + str(image.shape[0]-1))
        if st.button("Get RGB Value"):
            r, g, b = image_rgb[y, x]
            st.write(f"RGB Value at ({x}, {y}): ({r}, {g}, {b})")

            color_hex = f'#{r:02x}{g:02x}{b:02x}'
            st.markdown(f'<div style="width:100px;height:100px;background-color:{color_hex};"></div>', unsafe_allow_html=True)
            st.write(f"Color: {color_hex}")
            new_row = pd.DataFrame({'r': [r], 'g': [g], 'b': [b], 'color': [color_hex]})
            st.dataframe(new_row)

            # 在指定位置添加標記
            marked_image = image_rgb.copy()
            cv2.drawMarker(marked_image, (x, y), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, 
                           markerSize=10, thickness=2, line_type=cv2.LINE_AA)

            # 使用 PIL 將標記後的圖像轉換為可顯示的格式
            # marked_image_pil = Image.fromarray(marked_image)

            # 顯示標記後的圖像
            st.image(marked_image, caption="Marked Image", use_container_width=True)

            # color_value = st.number_input("Enter corresponding value")
            # st.write(f"Corresponding value: {color_value}")
            # st.button("Submit")

        # r = r if 'r' not in st.session_state else r
        # g = g if 'g' not in st.session_state else g
        # b = b if 'b' not in st.session_state else b
            # r
            # g
            # b
        # df_color = pd.DataFrame(columns=['r', 'g', 'b', 'color', 'value'])

            # if st.button("Submit"):

            #     if 'df_color' not in st.session_state:
            #         st.session_state.df_color = pd.DataFrame(columns=['r', 'g', 'b', 'color', 'value'])
            #     new_row = pd.DataFrame({'r': [r], 'g': [g], 'b': [b], 'color': [color_hex], 'value': [color_value]})
            #     st.session_state.df_color = pd.concat([st.session_state.df_color, new_row], ignore_index=True)
            #     # df_color = pd.DataFrame([[x, y, r, g, b, color_hex, color_value]], columns=['x', 'y', 'r', 'g', 'b', 'color', 'value']) 

            #     st.write("Submitted successfully.")
            # st.dataframe(st.session_state.df_color)

                
if __name__ == "__main__":
    main()