import streamlit as st
from openai import OpenAI
from google.cloud import bigquery
import pandas as pd
from google.oauth2 import service_account
import re
import plotly.express as px


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
# Initialize BigQuery client
bq_client = bigquery.Client(credentials=credentials, project='textsummarizerproject')


# Show title and description
st.title("💬 Text to SQL with Charts")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key and BigQuery table name
openai_api_key = st.text_input("OpenAI API Key", type="password")
bigquery_table_name = st.text_input("BigQuery Table Name (format: project.dataset.table)", placeholder="your-project.your-dataset.your-table")

def validate_table_name(table_name):
    """Validate BigQuery table name format"""
    if not table_name:
        return False, "Table name cannot be empty"
    
    parts = table_name.split('.')
    if len(parts) != 3:
        return False, f"Table name must have exactly 3 parts (project.dataset.table), found {len(parts)} parts"
    
    # Check for empty parts
    if any(not part.strip() for part in parts):
        return False, "Table name cannot have empty parts"
    
    return True, "Valid"

def format_currency(value):
    """Format numbers as currency with appropriate units"""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:,.2f}"

def create_chart1(data, query_info):
    """Create chart for aggregation data - show ALL aggregated results"""
    if data.empty or len(data.columns) < 2:
        return
    
    x_col, y_col = data.columns[0], data.columns[1]
    operation = query_info.get('operation', '').lower()
    
    # For aggregation queries, show ALL results since data is already summarized
    if operation in ['avg', 'average', 'mean']:
        fig = px.line(data, x=x_col, y=y_col, markers=True, 
                    title=f"Average {query_info['column']} by {query_info['group_by']}")
    else:
        fig = px.bar(data, x=x_col, y=y_col,
                    title=f"{operation.title()} of {query_info['column']} by {query_info['group_by']}")
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def create_chart(data, query_info):
    """Create chart for aggregation data"""
    if data.empty or len(data.columns) < 2:
        return
    
    x_col, y_col = data.columns[0], data.columns[1]
    operation = query_info.get('operation', '').lower()
    
    # Check if we need currency formatting
    needs_currency = any(keyword in query_info.get('column', '').lower() 
                        for keyword in ['balance', 'amount', 'price', 'revenue', 'sales', 'cost'])
    
    # Format the data for display
    display_data = data.copy()
    if needs_currency and pd.api.types.is_numeric_dtype(display_data[y_col]):
        display_data['formatted_value'] = display_data[y_col].apply(format_currency)
        
        # Create a formatted table
        st.subheader(f"💰 {operation.title()} of {query_info['column']} by {query_info['group_by']}")
        
        # Show formatted table
        formatted_table = display_data[[x_col, 'formatted_value']].copy()
        formatted_table.columns = [query_info['group_by'].title(), f"{operation.title()} {query_info['column'].title()}"]
        st.dataframe(formatted_table, use_container_width=True)
    else:
        st.subheader(f"📊 {operation.title()} of {query_info['column']} by {query_info['group_by']}")
        


if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
elif not bigquery_table_name:
    st.info("Please add your Google BigQuery Table Name to continue.")    
else:
    # Validate table name format
    is_valid, validation_message = validate_table_name(bigquery_table_name)
    
    if not is_valid:
        st.error(f"Invalid table name: {validation_message}")
        st.info("Example format: `my-project.my-dataset.my-table`")
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

        # Get row count with error handling
        row_count_query = f"SELECT COUNT(*) as total_rows FROM `{bigquery_table_name}`"

        try:
            row_count_df = bq_client.query(row_count_query).to_dataframe()
            total_rows = row_count_df.iloc[0]['total_rows']
            st.info(f"Total rows in table {bigquery_table_name} are : {total_rows:,}")
        except Exception as e:
            st.error(f"Error getting row count: {e}")

        # Safely parse table name parts
        try:
            parts = bigquery_table_name.split('.')
            project_id = parts[0]
            dataset_id = parts[1] 
            table_name = parts[2]
            
            columns_query = f"""
            SELECT column_name, data_type 
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            columns_df = bq_client.query(columns_query).to_dataframe()
            st.info(f"Below are the columns in table {bigquery_table_name} ")
            st.dataframe(columns_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error getting column information: {e}")

        

        def parse_query(prompt):
            """Parse user query and return query type and parameters"""
            prompt_lower = prompt.lower()
            
            # Check for column queries like "show columns", "list columns", "column names"
            if re.search(r'(?:show|list|get|display)\s+columns?|column\s+names?|what\s+columns?', prompt_lower):
                return {'type': 'columns'}
            
            # Check for row queries like "give me 10 rows"
            row_match = re.search(r'(\d+)\s*rows?', prompt_lower)
            if row_match:
                return {'type': 'rows', 'n_rows': int(row_match.group(1))}
            
            # Check for aggregation queries like "sum/avg/count/max/min of column by group"
            agg_match = re.search(r'(sum|average|avg|mean|count|max|min|total)\s+(?:of\s+)?(\w+)\s+by\s+(\w+)', prompt_lower)
            if agg_match:
                return {
                    'type': 'aggregation', 
                    'operation': agg_match.group(1), 
                    'column': agg_match.group(2), 
                    'group_by': agg_match.group(3)
                }
            
            return {'type': 'unknown'}

        # Create a chat input field
        if prompt := st.chat_input("Enter your query (e.g., 'Give me 10 rows' or 'Sum of sales by region')"):
            # Store and display the current prompt
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Parse the query
            query_info = parse_query(prompt)
            
            try:
                if query_info['type'] == 'rows':
                    # Generate SQL directly for row requests
                    QUERY = f"SELECT * FROM `{bigquery_table_name}` LIMIT {query_info['n_rows']}"

                elif query_info['type'] == 'columns':
                    # Generate SQL to get column names and types
                    parts = bigquery_table_name.split('.')
                    project_id = parts[0]
                    dataset_id = parts[1] 
                    table_name = parts[2]
                    QUERY = f"""
                    SELECT column_name, data_type 
                    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                    """

                elif query_info['type'] == 'aggregation':
                    # Generate SQL directly for aggregation requests
                    operation = query_info['operation'].upper()
                    if operation in ['AVERAGE', 'AVG', 'MEAN']:
                        operation = 'AVG'
                    elif operation == 'TOTAL':
                        operation = 'SUM'
                    
                    QUERY = f"SELECT {query_info['group_by']}, {operation}({query_info['column']}) as {operation.lower()}_{query_info['column']} FROM `{bigquery_table_name}` GROUP BY {query_info['group_by']}"
                    
                else:
                    # Use GPT for complex queries
                    stream = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"Generate only SQL query for table `{bigquery_table_name}`. No explanations."}
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
                
                try:
                    # Run BigQuery
                    bigquery_client = bigquery.Client.from_service_account_info(dict(st.secrets["gcp_service_account"]))
                    data = bigquery_client.query(QUERY).to_dataframe()
                    st.dataframe(data)
                    # Add chart for aggregation queries
                    if query_info['type'] == 'aggregation' and not data.empty:
                        create_chart(data, query_info)
                except Exception as e:
                    st.error(f"Error executing query: {e}")
                    # Try simple fallback
                    try:
                        fallback = f"SELECT * FROM `{bigquery_table_name}` LIMIT 10"
                        st.info("Trying fallback query...")
                        st.code(fallback)
                        data = bigquery_client.query(fallback).to_dataframe()
                        st.dataframe(data)
                    except Exception as fallback_error:
                        st.error(f"Fallback also failed: {fallback_error}")
                        
            except Exception as e:
                st.error(f"Error processing query: {e}")