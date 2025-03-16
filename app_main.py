import streamlit as st 

def main():
    st.title("Author & License:")
    st.markdown("**Kurt Su** (phononobserver@gmail.com)")
    st.markdown("**This tool release under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license**")

    st.markdown("               ")
    st.markdown("               ")

    st.header("Page Purpose & Description")
    st.markdown("**color_to_value**: Import image and transform color to value")
    st.markdown("**image_to_df**: Import image and extract color value to DataFrame & vtk file")
    st.markdown("**df_to_vtk**: Import dataframe, adjust coordiation and transform to vtk file")
    # st.markdown("**regression B ML**: Machine Learning regression/classification tool")
    # st.markdown("**predict**: Predict result with load trained model")
    # st.markdown("**predict performance**: Predict result accracy index with real result")

if __name__ == '__main__':
    main()
