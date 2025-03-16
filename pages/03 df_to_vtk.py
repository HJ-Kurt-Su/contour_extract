import numpy as np
import pandas as pd
import pyvista as pv
# import matplotlib.pyplot as plt
import streamlit as st
import datetime


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
    

def main():
    st.title('Color Value Extraction Tool')

    # uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    uploaded_file = st.sidebar.file_uploader('#### 選擇您要上傳的 Raw Data',type=["csv", "xlsx"], key="raw")
    # df_clr_value = upload_file(uploaded_color)
    # st.dataframe(df_clr_value)

    if uploaded_file is not None:
        # 將上傳的文件保存到本地
        # image_path = ".\pages\\" + uploaded_file.name
        df = upload_file(
            uploaded_raw=uploaded_file
        )

        st.dataframe(df)

        x_of_st = st.number_input("Enter x coordinate", value=0.0, step=0.1)
        y_of_st = st.number_input("Enter y coordinate", value=0.0, step=0.1)
        st.markdown("               ")
        st.markdown("Please consider the z coordinate scale level")
        st.markdown("Expect to excute scale then offset")
        z_scale = st.number_input("Enter z scale level", min_value=0.1, value=1.0, step=0.1)
        z_of_st = st.number_input("Enter z coordinate", value=0.0, step=0.1)

        df_adj = df.copy()
        if st.button("Adjust and Scale"):
        
            df_adj['x'] = df_adj['x'] + x_of_st
            df_adj['y'] = df_adj['y'] + y_of_st
            df_adj['z'] = df_adj['z'] * z_scale + z_of_st

            st.dataframe(df_adj)

        # x_pixel = df_merged['x'].max()
        # y_pixel = df_merged['y'].max()

        # df_merged['x'] = df_merged['x'] / x_pixel * x_dim
        # df_merged['y'] = df_merged['y'] / y_pixel * y_dim
        # df_merged['z'] = df_merged['z'] / 1000

        # 將 DataFrame 轉換為 VTK 文件
        # df_vtk = convert_df_to_vtk(df, "output.vtk")

        download_file(name_label="Input Result File Name",
                button_label='Download RGB as CSV',
                file=df_adj,
                file_type="csv",
                gui_key="rgb"
                )

        download_file(name_label="Input Result File Name",
                button_label='Download RGB as VTK',
                file=df_adj,
                file_type="vtk",
                gui_key="rgb_vtk"
                )

if __name__ == "__main__":
    main()