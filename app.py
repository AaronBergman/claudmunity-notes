import streamlit as st
import pandas as pd
import random
import anthropic
import requests
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Community Notes Assistant",
    page_icon="🗒️",
    layout="wide"
)

# Function to call Claude API
def call_claude_api(client, user_message, conversation_history=""):
    """Call the Anthropic Claude API using the Python client and return the assistant's response."""
    try:
        # Create messages list for the API
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            # Parse conversation history into proper message format
            history_lines = conversation_history.strip().split('\n')
            current_role = None
            current_content = []
            
            for line in history_lines:
                if line.startswith("User: "):
                    # If we were building a previous message, add it
                    if current_role and current_content:
                        messages.append({"role": current_role, "content": "\n".join(current_content)})
                        current_content = []
                    
                    current_role = "user"
                    current_content = [line[6:]]  # Remove "User: " prefix
                elif line.startswith("Assistant: "):
                    # If we were building a previous message, add it
                    if current_role and current_content:
                        messages.append({"role": current_role, "content": "\n".join(current_content)})
                        current_content = []
                    
                    current_role = "assistant"
                    current_content = [line[11:]]  # Remove "Assistant: " prefix
                elif line.startswith('-' * 80):
                    # Separator line, ignore
                    continue
                else:
                    # Continuation of current message
                    if current_role:
                        current_content.append(line)
            
            # Add the last message if there is one
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        # Use the Python client to create a message
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=1024,
            system="""You are an AI assistant trained to help expand the reach of Community Notes on X (formerly Twitter) by providing helpful, informative, and accurate context to posts that might be misleading or missing important context.

Your goal is to write notes that people from different points of view would find helpful. Focus on accuracy and factual information, providing sources when possible.

Key guidelines:
- Respond with "NNN" (No Note Needed) for posts that don't require additional context
- Do not correct obvious satire or jokes - respond with "NNN"
- Provide accurate, high-quality information with reliable sources
- Be informative and help users better understand the subject matter
- Write notes that would be helpful to people across different viewpoints
- Stay neutral and focus on facts rather than opinions
- Avoid partisan language or taking sides on controversial issues
- Only add context when it meaningfully improves understanding

If you're unsure about the accuracy of information, err on the side of caution.""",
            messages=messages
        )
        
        # Extract the text content from the response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            return "No response content received."
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")

# Main app function
def main():
    # Title and introduction
    st.title("Community Notes Assistant")
    st.markdown("""
    This app helps expand the reach of Community Notes by providing helpful context to posts that might be 
    misleading or missing important information. It uses Claude AI to generate informative responses.
    """)
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # API Key input
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your Anthropic API Key to use the app")
    
    # Number of context examples to load
    num_context_examples = st.sidebar.slider("Number of random examples", min_value=0, max_value=100, value=5)
    
    # Data source settings
    st.sidebar.header("Data Source")
    data_url = st.sidebar.text_input(
        "CSV Data URL", 
        value="https://f004.backblazeb2.com/file/aaronbergman-public/tweet_data_final_v1_filtered_2cols.csv"
    )
    
    # Button to load/reload data
    load_data = st.sidebar.button("Load/Reload Data")
    
    # Initialize session state for conversation history and chat
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ""
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize the DataFrame in session state
    if 'df' not in st.session_state or load_data:
        try:
            with st.spinner("Downloading and loading data..."):
                # Download and load the CSV file from URL
                response = requests.get(data_url)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Load the CSV data from the response content
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data, sep=None, engine='python')
                
                # Check if the column names match expected names
                if 'tweet_content' not in df.columns or 'summary' not in df.columns:
                    # Try to adapt to the actual column names if they exist but are named differently
                    if len(df.columns) >= 2:
                        df.columns = ['tweet_content', 'summary']
                        st.success(f"Renamed columns to 'tweet_content' and 'summary'")
                    else:
                        st.error(f"Error: Expected 'tweet_content' and 'summary' columns but found: {df.columns.tolist()}")
                        return
                
                # Filter out rows with empty or problematic content
                valid_df = df[
                    (df['tweet_content'].notna()) & 
                    (df['tweet_content'].str.strip() != '') &
                    (df['summary'].notna()) & 
                    (df['summary'].str.strip() != '')
                ].copy()
                
                st.session_state.df = valid_df
                st.success(f"Successfully loaded {len(valid_df)} valid rows from data")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.df = None
    
    # Load context examples if requested and data is available
    if st.session_state.df is not None and num_context_examples > 0:
        if st.sidebar.button("Generate New Context Examples"):
            with st.spinner(f"Selecting {num_context_examples} random examples..."):
                # Select random rows for context
                n = min(num_context_examples, len(st.session_state.df))
                random_indices = random.sample(range(len(st.session_state.df)), n)
                selected_rows = st.session_state.df.iloc[random_indices]
                
                # Generate conversation history from selected rows
                conversation_history = ""
                for _, row in selected_rows.iterrows():
                    tweet = row['tweet_content'].strip()
                    summary = row['summary'].strip()
                    
                    # Skip if either part is empty after stripping
                    if not tweet or not summary:
                        continue
                        
                    conversation_history += f"User: {tweet}\nAssistant: {summary}\n{'-' * 80}\n"
                
                st.session_state.conversation_history = conversation_history
                st.success(f"Generated {n} new context examples")
    
    # Display the context examples in a collapsible section
    if st.session_state.conversation_history:
        with st.expander("View Context Examples"):
            st.text(st.session_state.conversation_history)
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"**User:**\n{user_msg}")
            st.markdown(f"**Assistant:**\n{ai_msg}")
            st.markdown("---")
    
    # User input
    user_input = st.text_area("Enter your post or tweet:", height=100)
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Submit")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    if send_button and user_input and api_key:
        try:
            # Initialize the Anthropic client
            client = anthropic.Anthropic(api_key=api_key)
            
            # Get AI response using the conversation history
            with st.spinner("Generating response..."):
                ai_response = call_claude_api(client, user_input, st.session_state.conversation_history)
            
            # Add the exchange to chat history
            st.session_state.chat_history.append((user_input, ai_response))
            
            # Clear the input area and rerun to update the UI
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Warning if API key is missing
    if send_button and not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to use the chat functionality.")

if __name__ == "__main__":
    main()
