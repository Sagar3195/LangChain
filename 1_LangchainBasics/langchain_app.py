import os 
import streamlit as st 
from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_core.prompts import ChatPromptTemplate 

#Page config 
st.set_page_config(page_title= "Simple LangChain Chatbot with Groq", page_icon= "🚀")

#Title 
st.title("🚀 Simple LangChain Chatbot with Groq")

with st.sidebar:
    st.header("Settings")
    
    #API Key 
    api_key = st.text_input("GROQ API Key:", type="password")

    #Model Selection 
    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "gemma2-9b-it"],
        index = 0
    )

    #Clear Button 
    if st.button("Clear Chat"):
        st.session_state = []
        st.rerun()

    
#Initialize the chat history 
if "messages" not in st.session_state:
    st.session_state.messages  = []

#Initialize the LLM Model 
@st.cache_resource 
def get_chain(api_key, model_name):
    if not api_key:
        return None

    llm = ChatGroq(model= model_name, api_key= api_key, temperature= 0.7, streaming= True)

    #Create Prompt Template 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",  "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
            ("user", "{question}")
        ]
    )

    #Create chain 
    chain = prompt | llm | StrOutputParser()
    
    return chain 

#Get chain 
qa_chain = get_chain(api_key, model_name)

if not qa_chain:
    st.warning("👆 Please enter your Groq API key in the sidebar to start chatting!")
    st.markdown("[Get your free API key here](https://console.groq.com)")

else:
    #Display the chat messages 

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

     ## chat input
    if question:= st.chat_input("Ask me anything"):
        ## Add user message to session state
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response from Groq
                for chunk in qa_chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

## Examples

st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")


