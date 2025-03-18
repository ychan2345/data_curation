import streamlit as st
import pandas as pd
import json
from utils import process_uploaded_file

def get_confidence_indicator(confidence_score: float) -> str:
    """Return an emoji indicator based on confidence score"""
    if confidence_score >= 0.9:
        return "🟢"  # Green circle for high confidence
    elif confidence_score >= 0.8:
        return "🟡"  # Yellow circle for medium confidence
    else:
        return "🔴"  # Red circle for low confidence

def main():
    st.set_page_config(
        page_title="CSV Analysis Tool",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 CSV Analysis Tool with Azure OpenAI")

    # Initialize session state for API credentials
    if 'azure_api_key' not in st.session_state:
        st.session_state.azure_api_key = ''
    if 'azure_endpoint' not in st.session_state:
        st.session_state.azure_endpoint = ''

    # API credentials input section
    with st.expander("Azure OpenAI Credentials", expanded=True):
        api_key = st.text_input(
            "Enter your Azure OpenAI API Key:",
            type="password",
            value=st.session_state.azure_api_key
        )
        endpoint = st.text_input(
            "Enter your Azure OpenAI Endpoint:",
            value=st.session_state.azure_endpoint,
            placeholder="https://your-resource.openai.azure.com"
        )

    # File upload section - shown regardless of API credentials
    st.markdown("""
    Upload your CSV file to get detailed analysis and insights powered by Azure OpenAI.
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if not (api_key and endpoint):
        st.info("👆 Please enter your Azure OpenAI credentials to analyze the data!")
        return

    st.session_state.azure_api_key = api_key
    st.session_state.azure_endpoint = endpoint

    if uploaded_file is not None:
        try:
            with st.spinner("Processing your file..."):
                df, stats, analysis = process_uploaded_file(
                    uploaded_file,
                    api_key=api_key,
                    endpoint=endpoint
                )

            st.subheader("GPT-4 Generated Insights")

            # Dataset Description
            st.markdown("#### Description of the Dataset")
            st.write(analysis["dataset_description"])

            # Suggested Analysis
            st.markdown("#### Suggested Analysis")
            for suggestion in analysis["suggested_analysis"]:
                st.markdown(f"- {suggestion}")

            # Column Details Table with Export
            st.markdown("#### Column Details")
            if analysis["columns"]:
                column_data = {
                    'Column Name': [],
                    'Column Title': [],
                    'Data Type': [],
                    'Confidence Score': [],
                    'Column Description': []
                }

                for col in analysis['columns']:
                    column_name = col['name']
                    column_data['Column Name'].append(column_name)
                    column_data['Column Title'].append(col['title'])
                    try:
                        dtype = str(df[column_name].dtype) if column_name in df.columns else 'N/A'
                    except:
                        dtype = 'Unknown'
                    column_data['Data Type'].append(dtype)
                    confidence = col.get('confidence_score', 0)
                    column_data['Confidence Score'].append(f"{get_confidence_indicator(confidence)} {confidence:.2%}")
                    column_data['Column Description'].append(col['description'])

                # Display the dataframe
                column_df = pd.DataFrame(column_data)
                st.dataframe(column_df, use_container_width=True)

                # Export Column Details
                st.download_button(
                    "Export Column Details (CSV)",
                    column_df.to_csv(index=False),
                    "column_details.csv",
                    "text/csv"
                )

            # Key Observations
            st.markdown("#### Key Observations")
            for observation in analysis["key_observations"]:
                st.markdown(f"- {observation}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Please ensure your CSV file is properly formatted and try again.")

if __name__ == "__main__":
    main()