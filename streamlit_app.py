import streamlit as st
from openai import OpenAI
from google.cloud import bigquery

# Show title and description.
st.title("💬 My new Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key and BigQuery table name
openai_api_key = st.text_input("OpenAI API Key", type="password")
bigquery_table_name = st.text_input("BigQuery Table Name", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
elif not bigquery_table_name:
    st.info("Please add your Google BigQuery Table Name to continue.")    
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display the existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field
    if prompt := st.chat_input("Get me first 10 rows"):
        # Store and display the current prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate SQL query
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Generate only SQL query for table {bigquery_table_name}. No explanations."}
            ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Clean SQL query
        QUERY = response.replace("```sql", "").replace("```", "").strip()
        
        # Show what we're running
        st.code(QUERY)
        bigquery_client = bigquery.Client.from_service_account_info(dict(st.secrets["gcp_service_account"]))
        try:
            # Run BigQuery
            
            data = bigquery_client.query(QUERY).to_dataframe()
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error: {e}")
            # Try simple fallback
            try:
                fallback = f"SELECT * FROM {bigquery_table_name} LIMIT 10"
                data = bigquery_client.query(fallback).to_dataframe()
                st.dataframe(data)
            except Exception as fallback_error:
                st.error(f"Fallback also failed: {fallback_error}")