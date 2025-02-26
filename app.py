import streamlit as st
import pandas as pd
import random
import os
import anthropic
import requests
from io import StringIO

# Set page config for a nicer appearance
st.set_page_config(
    page_title="Community Notes Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e3f6f5;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        width: 100%;
    }
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    .stTextInput>div>div>input {
        padding: 0.75rem;
    }
    hr {
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to call Claude API
def call_claude_api(client, user_message):
    """Call the Anthropic Claude API and return the assistant's response."""
    try:
        messages = [{"role": "user", "content": user_message}]
        
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
        
        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            return "No response content received."
    except Exception as e:
        return f"API request failed: {str(e)}"

# Create a function to load data
@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_data(url):
    try:
        with st.spinner("Downloading and processing data..."):
            response = requests.get(url)
            response.raise_for_status()
            
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, sep=None, engine='python')
            
            # Check and rename columns if needed
            if 'tweet_content' not in df.columns or 'summary' not in df.columns:
                if len(df.columns) >= 2:
                    df.columns = ['tweet_content', 'summary']
                else:
                    st.error(f"Expected 'tweet_content' and 'summary' columns but found: {df.columns.tolist()}")
                    return None
            
            # Filter out rows with empty content
            valid_df = df[
                (df['tweet_content'].notna()) & 
                (df['tweet_content'].str.strip() != '') &
                (df['summary'].notna()) & 
                (df['summary'].str.strip() != '')
            ].copy()
            
            return valid_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'client' not in st.session_state:
    st.session_state.client = None

if 'df' not in st.session_state:
    st.session_state.df = None

if 'samples_generated' not in st.session_state:
    st.session_state.samples_generated = False

# Function to display chat message
def display_message(role, content, avatar=None):
    col1, col2 = st.columns([1, 9])
    
    with col1:
        if avatar:
            st.image(avatar, width=50)
        elif role == "user":
            st.markdown("👤")
        else:
            st.markdown("🤖")
    
    with col2:
        st.markdown(f"**{role.title()}**: {content}")
    
    st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("Community Notes Assistant")
    st.markdown("This app helps provide context for potentially misleading content using Claude API.")
    
    # Data source settings
    st.subheader("Data Settings")
    data_url = st.text_input(
        "CSV Data URL", 
        "https://f004.backblazeb2.com/file/aaronbergman-public/tweet_data_final_v1_filtered_2cols.csv",
        help="URL to the CSV file containing tweet content and summaries"
    )
    
    sample_count = st.number_input("Number of samples to show", min_value=1, max_value=100, value=5)
    
    if st.button("Load Data & Generate Samples"):
        # Load the data
        st.session_state.df = load_data(data_url)
        if st.session_state.df is not None:
            st.success(f"Loaded {len(st.session_state.df)} valid rows of data")
            st.session_state.samples_generated = True
        else:
            st.error("Failed to load data")
    
    # API settings
    st.subheader("API Settings")
    api_key_input = st.text_input("Anthropic API Key", type="password", 
                                help="Your Anthropic API key for Claude. Will be stored in session state only.")
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        try:
            st.session_state.client = anthropic.Anthropic(api_key=st.session_state.api_key)
            st.success("API Key set successfully")
        except Exception as e:
            st.error(f"Error setting API key: {e}")
    
    # About section
    st.subheader("About")
    st.markdown("""
    This application demonstrates using Claude API to analyze content and provide Community Notes-style context.
    
    Created with Streamlit. [Source code available](https://github.com/yourusername/community-notes-assistant).
    """)

# Main content area
st.title("Community Notes Assistant")

# Tabs for different functionality
tab1, tab2 = st.tabs(["Interactive Chat", "Sample Notes"])

# Tab 1: Interactive Chat
with tab1:
    st.header("Chat with the Community Notes Assistant")
    st.markdown("""
    Type a tweet or post to get a Community Notes-style response from Claude. 
    The assistant will provide helpful context for potentially misleading content, 
    or reply with "NNN" (No Note Needed) if additional context isn't necessary.
    """)
    
    # Display previous messages
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])
    
    # User input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=100, 
                                 placeholder="Enter a tweet or post to analyze...")
        submit_button = st.form_submit_button("Send")
    
    # Process user input and get response
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message("user", user_input)
        
        # Get response from Claude if API key is set
        if st.session_state.client:
            with st.spinner("Getting response from Claude..."):
                response = call_claude_api(st.session_state.client, user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_message("assistant", response)
        else:
            st.warning("Please set your Anthropic API key in the sidebar to get responses")
    
    # Add a button to clear the chat
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Tab 2: Sample Notes
with tab2:
    st.header("Sample Community Notes")
    
    if st.session_state.samples_generated and st.session_state.df is not None:
        df = st.session_state.df
        
        # Select random samples
        n = min(sample_count, len(df))
        random_indices = random.sample(range(len(df)), n)
        selected_rows = df.iloc[random_indices]
        
        # Display samples
        for i, (_, row) in enumerate(selected_rows.iterrows()):
            tweet = row['tweet_content'].strip()
            summary = row['summary'].strip()
            
            if not tweet or not summary:
                continue
            
            st.markdown(f"### Sample {i+1}")
            display_message("user", tweet)
            display_message("assistant", summary)
    else:
        st.info("Click 'Load Data & Generate Samples' in the sidebar to view random samples")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Claude | © 2025")