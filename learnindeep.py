import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Learning prompt template
LEARNING_SYSTEM_PROMPT = """You are an exceptional, patient, and dedicated professor with years of teaching experience. Your mission is to take students from zero knowledge to advanced proficiency in their chosen subject.

## Core Teaching Principles:

1. **Start with Foundations**: Always begin with the absolute basics, assuming minimal prior knowledge. Build concepts layer by layer.

2. **Explain the "Why"**: For every concept, technique, or approach:
   - Explain WHY it exists and what problem it solves
   - Provide historical context or motivation
   - Show what happens if we DON'T use this approach

3. **Present Alternatives**: For each concept or method:
   - Discuss alternative approaches
   - Compare pros and cons
   - Explain when to use which approach
   - Mention what people used before this technique existed

4. **Use Progressive Complexity**: 
   - Start simple, then gradually increase difficulty
   - Use analogies and real-world examples
   - Provide code examples that build on previous ones
   - Connect new concepts to previously learned ones

5. **Interactive Learning**:
   - After explaining concepts, provide practice exercises
   - Check understanding before moving forward
   - Encourage questions at any point
   - Adapt pace based on student responses

6. **Complete Curriculum Structure**:
   - Create a clear roadmap of topics
   - Number lessons/modules logically
   - Show how topics interconnect
   - Indicate prerequisites for advanced topics

7. **Practical Application**:
   - Include hands-on coding examples
   - Provide real-world use cases
   - Discuss industry best practices
   - Mention common pitfalls and how to avoid them

8. **Reasoning and Critical Thinking**:
   - Emphasize "WHY" not just "what" and "how"
   - Explain reasoning behind design decisions
   - Discuss trade-offs in different approaches
   - Help develop intuition, not just memorization

## Your Teaching Style:
- Be patient, encouraging, and approachable
- Use clear, jargon-free language (explain jargon when necessary)
- Break complex topics into digestible chunks
- Provide summaries after major sections
- Use analogies from everyday life
- Be enthusiastic about the subject matter

## Lesson Format:
Each lesson should include:
- Lesson Title
- Objectives
- Prerequisites
- Concept Introduction
- The Reasoning
- Alternatives
- Hands-On Example
- Common Mistakes
- Practice Exercise
- Summary
- Next Steps"""

# Page configuration
st.set_page_config(
    page_title="Learning Chatbot",
    page_icon="📚"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "started" not in st.session_state:
    st.session_state.started = False
if "llm" not in st.session_state:
    st.session_state.llm = None

# Title
st.title("📚 Learning in Depth Chatbot")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", "")
    )
    
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.divider()
    
    topic = st.text_input("What do you want to learn?", value=st.session_state.topic)
    
    knowledge_level = st.selectbox(
        "Your Knowledge Level",
        ["Complete Beginner", "Some Basics", "Intermediate", "Advanced"]
    )
    
    if st.button("Start Learning", type="primary"):
        if not api_key:
            st.error("Please enter API key")
        elif not topic:
            st.error("Please enter a topic")
        else:
            try:
                st.session_state.topic = topic
                st.session_state.messages = []
                st.session_state.started = True
                
                # Initialize LLM
                st.session_state.llm = ChatGroq(
                    model=model,
                    temperature=temperature,
                    groq_api_key=api_key
                )
                
                # Create initial messages
                system_msg = SystemMessage(content=LEARNING_SYSTEM_PROMPT)
                initial_msg = HumanMessage(content=f"""I want to learn {topic}. My current knowledge level is: {knowledge_level}. 

Please:
1. Introduce yourself as my dedicated professor
2. Provide a complete curriculum overview/roadmap for learning {topic}
3. Start with Lesson 1 when ready""")
                
                st.session_state.messages = [system_msg, initial_msg]
                
                # Get initial response
                response = st.session_state.llm.invoke(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=response.content))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "invalid_api_key" in str(e).lower() or "401" in str(e):
                    st.error("Invalid API Key! Please get a valid key from: https://console.groq.com/keys")
                st.session_state.started = False
                st.session_state.llm = None
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.topic = ""
        st.session_state.started = False
        st.session_state.llm = None
        st.rerun()

# Main chat area
if st.session_state.started:
    # Display messages
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Show user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Get AI response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.llm.invoke(st.session_state.messages)
                    st.write(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if "invalid_api_key" in str(e).lower() or "401" in str(e):
                st.error("Invalid API Key! Please check your key in the sidebar.")
else:
    st.info("👈 Enter a topic in the sidebar and click 'Start Learning' to begin!")
    
