import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import streamlit as st
import datetime
import pyvista as pv
from scipy.interpolate import griddata

# from pyvista import Plotter

# from streamlit_drawable_canvas import st_canvas


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def upload_file(uploaded_raw):

    if uploaded_raw is not None:
        up_file_type = uploaded_raw.name.split(".")[1]

        if up_file_type == "csv":
            df_raw = pd.read_csv(uploaded_raw, encoding="utf-8")
        elif up_file_type == "xlsx":
            df_raw = pd.read_excel(uploaded_raw)
        st.header('您所上傳的CSV檔內容：')

        return df_raw
    

def extract_color_values(image_path, rgb_white_fil):
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")
    else:
        st.write("Image found and read successfully")
        st.image(image_path, caption="Uploaded Image", use_container_width=True)

    # 將圖像轉換為 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # st.image(image_rgb, caption="RGB Image", use_container_width=True)

    # 提取顏色值
    color_data = []
    for y in range(image_rgb.shape[0]):
        for x in range(image_rgb.shape[1]):
            r, g, b = image_rgb[y, x]
            if not (r >= rgb_white_fil and g >= rgb_white_fil and b >= rgb_white_fil):
                color_data.append([x, y, r, g, b])
       
    color_data = np.array(color_data)
    # color_data 

    # st.write("Click on the image to get RGB values and specify corresponding value")
    # clicked = st.image(image_path, use_container_width=True)
    # if clicked:
    #     x = st.number_input("Enter x coordinate", min_value=0, max_value=image.shape[1]-1)
    #     y = st.number_input("Enter y coordinate", min_value=0, max_value=image.shape[0]-1)
    #     if st.button("Get RGB Value"):
    #         r, g, b = image_rgb[y, x]
    #         st.write(f"RGB Value at ({x}, {y}): ({r}, {g}, {b})")
    #         value = st.number_input("Enter corresponding value")
    #         st.write(f"Corresponding value: {value}")

    # 轉換為 DataFrame
    df = pd.DataFrame(color_data, columns=['x', 'y', 'r', 'g', 'b'])
    df.shape
    return df


def interpolate_values(df, df_clr_value):
    # 提取已知的 RGB 和對應的數值
    known_points = df_clr_value[['r', 'g', 'b']].values
    known_values = df_clr_value['value'].values

    # 提取需要內插的 RGB
    interp_points = df[['r', 'g', 'b']].values
    # interp_points
    # # interp_points.shape
    # interp_points.shape[0]
    # interp_points.shape[1]


    # 進行內插
    interp_values = griddata(known_points, known_values, interp_points, method='linear')
    # interp_values.shape[0]
    # interp_values.shape[1]
    

    missing_values = np.isnan(interp_values)
    if np.any(missing_values):
        interp_values[missing_values] = griddata(known_points, known_values, interp_points[missing_values], method='nearest')
    # interp_values = interp_values.flatten()

    # 將內插結果添加到 DataFrame
    df['value'] = interp_values
    return df

def convert_df_to_vtk(df, output_path):
    # 創建一個 PyVista 的 PolyData 對象
    points = df[['x', 'y', 'z']].values
    # points = np.c_[points, np.zeros(points.shape[0])]
    polydata = pv.PolyData(points)

    # 添加顏色數據

    polydata['r'] = df['r'].values
    polydata['g'] = df['g'].values
    polydata['b'] = df['b'].values
    polydata['value'] = df['value'].values
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
        output_path = "output_result.vtk"
        polydata = convert_df_to_vtk(file, output_path)
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
    uploaded_color = st.sidebar.file_uploader('#### 選擇您要上傳的 Color Value',type=["csv", "xlsx"], key="raw")
    df_clr_value = upload_file(uploaded_color)
    st.dataframe(df_clr_value)

    x_dim = st.number_input("Enter x dimension", min_value=1.0, value=100.0, step=0.1)
    y_dim = st.number_input("Enter y coordinate", min_value=1.0, value=168.0, step=0.1)


    if uploaded_file is not None:
        # 將上傳的文件保存到本地
        # image_path = ".\pages\\" + uploaded_file.name
        image_path = uploaded_file.name
        st.write(image_path)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 提取顏色值並顯示 DataFrame
        rgb_white_fil = st.number_input("Enter white filter", min_value=0, max_value=255, value=200)
        df = extract_color_values(image_path, rgb_white_fil)
        st.dataframe(df)

        # 合併兩個 DataFrame 並進行內插
        if df_clr_value is not None:
            df_merged = interpolate_values(df, df_clr_value)
            df_merged['z'] = df_merged['value']
            st.dataframe(df_merged)

        x_pixel = df_merged['x'].max()
        y_pixel = df_merged['y'].max()

        df_merged['x'] = df_merged['x'] / x_pixel * x_dim
        df_merged['y'] = df_merged['y'] / y_pixel * y_dim
        df_merged['z'] = df_merged['z'] / 1000

        # 將 DataFrame 轉換為 VTK 文件
        # df_vtk = convert_df_to_vtk(df, "output.vtk")

        download_file(name_label="Input Result File Name",
                button_label='Download RGB as CSV',
                file=df_merged,
                file_type="csv",
                gui_key="rgb"
                )

        download_file(name_label="Input Result File Name",
                button_label='Download RGB as VTK',
                file=df_merged,
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