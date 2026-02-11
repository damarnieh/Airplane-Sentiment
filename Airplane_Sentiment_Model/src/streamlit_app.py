import streamlit as st


with st.sidebar:
    st.title("Navigation")
    selection = st.radio(
        "Go to Page",
        ["EDA", "Classification"]
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        This application is used to analyze **text sentiment**.

        - **EDA**: Explore and understand the sentiment dataset  
        - **Classification**: Classify sentiment into  
          **Negative, Neutral, or Positive**
        """
    )


if selection == "EDA":
    import eda
    eda.run()

elif selection == "Classification":
    import predict
    predict.run()
