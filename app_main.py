import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import pyvista as pv
from pyvista import Plotter

# from streamlit_drawable_canvas import st_canvas


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def extract_color_values(image_path):
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")
    else:
        st.write("Image found and read successfully")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # 將圖像轉換為 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 提取顏色值
    color_data = []
    for y in range(image_rgb.shape[0]):
        for x in range(image_rgb.shape[1]):
            r, g, b = image_rgb[y, x]
            color_data.append([x, y, r, g, b])

    st.write("Click on the image to get RGB values and specify corresponding value")
    clicked = st.image(image_path, use_container_width=True)
    if clicked:
        x = st.number_input("Enter x coordinate", min_value=0, max_value=image.shape[1]-1)
        y = st.number_input("Enter y coordinate", min_value=0, max_value=image.shape[0]-1)
        if st.button("Get RGB Value"):
            r, g, b = image_rgb[y, x]
            st.write(f"RGB Value at ({x}, {y}): ({r}, {g}, {b})")
            value = st.number_input("Enter corresponding value")
            st.write(f"Corresponding value: {value}")
    # 轉換為 DataFrame
    df = pd.DataFrame(color_data, columns=['x', 'y', 'r', 'g', 'b'])
    return df, image_rgb


def convert_df_to_vtk(df, output_path):
    # 創建一個 PyVista 的 PolyData 對象
    points = df[['x', 'y']].values
    points = np.c_[points, np.zeros(points.shape[0])]
    polydata = pv.PolyData(points)

    # 添加顏色數據
    polydata['r'] = df['r'].values
    polydata['g'] = df['g'].values
    polydata['b'] = df['b'].values
    polydata.save(output_path)

    # return polydata

    # 保存為 VTK 文件
   


# def display_vtk(polydata):
#     plotter = Plotter(off_screen=True)
#     plotter.add_mesh(polydata, scalars='r', rgb=True)
#     plotter.show(screenshot='screenshot.png')
#     st.image('screenshot.png')

def download_file(name_label, button_label, file, file_type, gui_key):
    date = str(datetime.datetime.now()).split(" ")[0]
    if file_type == "csv":
        file = convert_df(file)
        mime_text = 'text/csv'
    # elif file_type == "html":
    #     file = convert_fig(file)  
    #     mime_text = 'text/html'
    elif file_type == "vtk":
        output_path = f"{date}_result.vtk"
        # polydata = convert_df_to_vtk(file, output_path)
        with open(output_path, 'rb') as f:
            file = f.read()
        mime_text = 'application/octet-stream'
        # display_vtk(polydata)

    # file_name_col, button_col = st.columns(2)
    result_file = date + "_result"
    download_name = st.text_input(label=name_label, value=result_file, key=gui_key) 
    download_name = download_name + "." + file_type

    st.download_button(label=button_label,  
                    data=file, 
                    file_name=download_name,
                    mime=mime_text,
                    key=gui_key+"dl")

# def extract_legend_colors(image_path, legend_coords):
#     # 讀取圖像
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or unable to read")

#     # 提取圖例區域
#     legend = image[legend_coords[1]:legend_coords[3], legend_coords[0]:legend_coords[2]]

#     # 將圖例轉換為 RGB 格式
#     legend_rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)

#     # 提取顏色值
#     unique_colors = np.unique(legend_rgb.reshape(-1, legend_rgb.shape[2]), axis=0)

#     # 假設圖例顏色從上到下對應的數值
#     values = np.linspace(1, 0, len(unique_colors))  # 這裡假設數值從 1 到 0

#     # 創建顏色對應的數值 DataFrame
#     color_value_df = pd.DataFrame(unique_colors, columns=['r', 'g', 'b'])
#     color_value_df['value'] = values

#     return color_value_df, 


def main():
    st.title('Color Value Extraction Tool')

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 將上傳的文件保存到本地
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 提取顏色值並顯示 DataFrame
        df, image_rgb = extract_color_values(image_path)
        st.dataframe(df)

        # 將 DataFrame 轉換為 VTK 文件
        # df_vtk = convert_df_to_vtk(df, "output.vtk")

        download_file(name_label="Input Result File Name",
                button_label='Download RGB as CSV',
                file=df,
                file_type="csv",
                gui_key="rgb"
                )

        download_file(name_label="Input Result File Name",
                button_label='Download RGB as VTK',
                file=df,
                file_type="vtk",
                gui_key="rgb_vtk"
                )


        # canvas_result = st_canvas(
        #     fill_color="rgba(255, 165, 0, 0.3)",
        #     stroke_width=2,
        #     stroke_color="#000",
        #     background_image=Image.open(image_path),
        #     update_streamlit=True,
        #     height=image_rgb.shape[0],
        #     width=image_rgb.shape[1],
        #     drawing_mode="freedraw",
        #     key="canvas",
        # )

        # if canvas_result.json_data is not None:
        #     for obj in canvas_result.json_data["objects"]:
        #         if obj["type"] == "circle":
        #             x = int(obj["left"])
        #             y = int(obj["top"])
        #             r, g, b = image_rgb[y, x]
        #             st.write(f"RGB Value at ({x}, {y}): ({r}, {g}, {b})")
        #             value = st.number_input("Enter corresponding value")
        #             st.write(f"Corresponding value: {value}")


        # legend_coords = (0, 0, 50, 256)  # 假設圖例在圖像的左側，這裡需要根據實際情況調整
        # color_value_df = extract_legend_colors(image_path, legend_coords)
        # st.dataframe(color_value_df)

        # st.write("Click on the image to get RGB values and specify corresponding value")
        # clicked = st.image(image_path, use_column_width=True)
        # if clicked:
        #     x = st.number_input("Enter x coordinate", min_value=0, max_value=image.shape[1]-1)
        #     y = st.number_input("Enter y coordinate", min_value=0, max_value=image.shape[0]-1)
        #     if st.button("Get RGB Value"):
        #         r, g, b = image_rgb[y, x]
        #         st.write(f"RGB Value at ({x}, {y}): ({r}, {g}, {b})")
        #         value = st.number_input("Enter corresponding value")
        #         st.write(f"Corresponding value: {value}")

        # 顯示顏色值的直方圖
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].hist(df['r'], bins=256, color='red', alpha=0.5)
        # ax[0].set_title('Red Channel')
        # ax[1].hist(df['g'], bins=256, color='green', alpha=0.5)
        # ax[1].set_title('Green Channel')
        # ax[2].hist(df['b'], bins=256, color='blue', alpha=0.5)
        # ax[2].set_title('Blue Channel')
        # st.pyplot(fig)

if __name__ == "__main__":
    main()