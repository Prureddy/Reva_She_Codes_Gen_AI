from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time



col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("https://github.com/Prureddy/MedChat/assets/100853494/86b9efcc-32cd-42ae-a2ce-90e5c6c0401e")

st.markdown(
    """
    
    <style>
div.stButton > button:first-child {
    background-color: #ffd0d0;
}
div.stButton > button:active {
    background-color: #ff6262;
}
#css files

   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()
    
if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate. Consult a qualified doctor for accurate advice._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True, "revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("medchat_db", embeddings)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """<s>[INST]Follow these instructions carefully: You are a medical practitioner chatbot providing accurate medical information, adopting a doctor's perspective in your responses. Utilize the provided context, chat history, and question, choosing only the necessary information based on the user's query. Avoid generating your own questions and answers. Do not reference chat history if irrelevant to the current question; only use it for similar-related queries. Prioritize the given context when searching for relevant information, emphasizing clarity and conciseness in your responses. If multiple medicines share the same name but have different strengths, ensure to mention them. Exclude any mention of medicine costs. Stick to context directly related to the user's question, and use your knowledge base to answer inquiries outside the given context. Abstract and concise responses are key; do not repeat the chat template in your answers. If you lack information, simply state that you don't know.

CONTEXT: {context}

CHAT HISTORY: {chat_history}

QUESTION: {question}

ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

TOGETHER_AI_API= os.environ['TOGETHER_AI']
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=512,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")
import streamlit as st
import pyttsx3
import speech_recognition as sr
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_option_menu import option_menu
from streamlit_chat import message
from utils import *
from googletrans import Translator
import os
import google.generativeai as genai
from PIL import Image
import json
from dotenv import load_dotenv


# Voice assistant setup
def speak(text, language='en'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('voice', f'com.apple.speech.synthesis.voice.{language}')
    engine.say(text)
    engine.runAndWait()


def listen(language='en'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language=language)
        print(f"User said: {query}\n")
    except Exception as e:
        print("Say that again please...")
        query = None
    return query


# Main app setup
def main():
    st.set_page_config(page_title="PetCareMate")
    st.header('Welcome to :violet[PetCareMate]')
    # Add translation language selection dropdown


    class AHome:
        @staticmethod
        def app():

            target_language = st.selectbox("Select Translation Language",
                                           ['en', 'kn', 'te', 'hi', 'bn''mr', 'ta', 'ur', 'gu', 'ml'])  # Add more languages as needed
            try:
                if 'responses' not in st.session_state:
                    st.session_state['responses'] = ["Hello!, How can I assist you?"]

                if 'requests' not in st.session_state:
                    st.session_state['requests'] = []

                with open('config.json') as config_file:
                    config = json.load(config_file)
                    api_key = config.get('openai_api_key', None)

                if api_key is None:
                    st.error("API key is missing. Please provide a valid API key in the config.json file.")
                    st.stop()
                else:
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

                    if 'buffer_memory' not in st.session_state:
                        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=5, return_messages=True)
            except ValueError:
                st.error("Incorrect API key provided. Please check and update the API key in the config.json file.")
                st.stop()
            except Exception as e:
                st.error("An error occurred during initialization: " + str(e))
                llm = None
                st.session_state.buffer_memory = None

            # Rest of your Streamlit app code...

            system_msg_template = SystemMessagePromptTemplate.from_template(
                template="""Please provide helpful information related to pet care and well-being based on your knowledge base. If the question is apart from pet care, please respond with 'I'm sorry, but I'm not equipped to answer that question at the moment. My expertise lies within certain domains, and this topic falls outside of those areas. However, I'm here to assist you with any inquiries within my capabilities. Is there anything else I can help you with?'
    '"""
            )

            human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

            prompt_template = ChatPromptTemplate.from_messages(
                [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

            conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm,
                                             verbose=True)

            # Container for chat history
            response_container = st.container()
            # Container for text box
            textcontainer = st.container()

            with textcontainer:
                query = st.text_input("Query: ", key="input")
                # Voice assistant integration
                if st.button("🎙️ Speak"):
                    user_query = listen(language='kn')  # Default language is English
                    if user_query:
                        st.text_area("User Query:", user_query)
                        # Process the user query and generate a response
                        response = conversation.predict(input=user_query)
                        st.text_area("Assistant Response:", response)
                        speak(response, language='kn')  # Output in English
                    else:
                        st.text_area("User Query:", "No query detected.")

                # Within the block where you handle user queries and generate responses
                if query:
                    with st.spinner("typing..."):
                        try:
                            conversation_string = get_conversation_string()
                            refined_query = query_refiner(conversation_string, query)
                            st.subheader("Refined Query:")
                            st.write(refined_query)
                            context = find_match(query)

                            if "pet" in context.lower() or "dog" in context.lower() or "cat" in context.lower() or "health" in context.lower() or "care" in context.lower() or "Training" in context.lower() or "nutrition" in context.lower():
                                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

                                def translate_text(text, target_language):
                                    translator = Translator()
                                    translated_text = translator.translate(text, dest=target_language).text
                                    return translated_text

                                try:
                                    response_translated = translate_text(response, target_language)
                                    st.session_state.requests.append(query)
                                    st.session_state.responses.append(response_translated)
                                except Exception as e:
                                    st.error("An error occurred during translation: " + str(e))
                            else:
                                st.warning(
                                    "I'm sorry, but I'm not equipped to answer that question                                    at the moment. My expertise lies within certain domains, and this topic falls outside of those areas. However, I'm here to assist you with any inquiries within my capabilities. Is there anything else I can help you with?")

                        except Exception as e:
                            st.error("An error occurred during conversation: " + str(e))

            with response_container:
                if st.session_state['responses']:
                    for i in range(len(st.session_state['responses'])):
                        try:
                            message(st.session_state['responses'][i], key=str(i))
                            if i < len(st.session_state['requests']):
                                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
                        except Exception as e:
                            st.error("An error occurred while displaying messages: " + str(e))

    class AboutUs:
        @staticmethod
        def app():
            html_content = """
                <style>
                    .header {
                        color: #4CAF50;
                    }
                    .subheader {
                        color: #008CBA;
                    }
                    .highlight {
                        color: #FF9800;
                        font-weight: bold;
                    }
                </style>
                <h1 class="header">Generative AI in PetCare</h1>

                <p>Generative AI, a subset of artificial intelligence (AI), is revolutionizing pet healthcare by providing advanced capabilities to improve various aspects of pet care. By leveraging generative AI technologies, PetCareMate aims to transform the way pet owners interact with healthcare information and services.</p>

                <h2 class="subheader">Project Overview</h2>

                <p>PetCareMate is an innovative platform leveraging the power of generative AI to transform pet healthcare. Our comprehensive solution encompasses disease detection, AI-driven chatbot support, multilingual capabilities, language translation, education, and empowerment for pet owners.</p>
                <ul>
                    <li><span class="highlight">Disease Detection:</span> Utilizing machine learning algorithms to analyze symptoms and diagnose pet diseases accurately.</li>
                    <li><span class="highlight">AI-driven Chatbot Support:</span> Offering personalized assistance to pet owners through intelligent chatbots powered by AI.</li>
                    <li><span class="highlight">Multilingual Capabilities:</span> Providing support for multiple languages to cater to diverse pet owner demographics.</li>
                    <li><span class="highlight">Language Translation:</span> Facilitating communication between pet owners and healthcare providers by translating information into different languages.</li>
                    <li><span class="highlight">Education:</span> Empowering pet owners with knowledge about pet healthcare, nutrition, training, and overall well-being.</li>
                    <li><span class="highlight">Empowerment:</span> Equipping pet owners with tools and resources to make informed decisions and actively participate in their pets' healthcare journey.</li>
                </ul>

                <h2 class="subheader">Future Enhancement</h2>

                <ol>
                    <li>
                        <p class="highlight">Telemedicine and Remote Monitoring:</p>
                        <p>Expand the platform to support telemedicine consultations with veterinarians, allowing users to connect with veterinary professionals remotely for advice, diagnosis, and treatment planning. Additionally, integrate IoT (Internet of Things) devices for remote monitoring of animal vital signs and health parameters, providing real-time insights to caregivers and veterinarians.</p>
                    </li>
                    <li>
                        <p class="highlight">Community-driven Data Collection and Analysis:</p>
                        <p>Implement features that allow users to contribute anonymized data about their animals' health conditions and treatments. Aggregate this data to identify trends, emerging diseases, and regional health disparities, enabling proactive public health interventions and targeted veterinary services.</p>
                    </li>
                    <li>
                        <p class="highlight">Fundraising and Revenue Generation:</p>
                        <p>Explore revenue generation strategies such as subscription models offering tiered access to premium features, partnerships with pet healthcare companies, and sponsored content.</p>
                    </li>
                </ol>

                <h2 class="subheader">Contact Us</h2>

                <p>For any inquiries or suggestions, feel free to reach out to us at <a href="mailto:contact@email.com">contact@email.com</a>. We value your feedback and strive to continuously improve PetCareMate to better serve the pet care community.</p>
            """

            st.write(html_content, unsafe_allow_html=True)

    class GeminiHealthApp:
        def __init__(self):
            load_dotenv()  ## load all the environment variables
            self.api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=self.api_key)
            self.input_prompt = """ As a pet healthcare expert, your role is to identify symptoms of pet diseases from provided descriptions or images. You should then provide detailed information on the disease, including its causes, symptoms, prevention measures, and recommended medications if necessary.
                                    Your response should follow this format:
                                    1. **Disease Name:**
                                       - **Symptoms:** Brief explanation of common symptoms.
                                       - **Causes:** Overview of possible causes.
                                       - **Prevention Measures:** Recommendations to prevent the disease.
                                       - **Recommended Medications:** If applicable, suggest appropriate medications.

                                    Please ensure your responses are comprehensive and aimed at helping pet owners understand and manage their pet's health effectively. Remember to include a disclaimer at the end of your response stating that you are not a veterinarian, and encourage users to seek professional advice.
                                    """

        def get_gemini_response(self, input_text, image, prompt):
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([input_text, image[0], prompt])
            return response.text

        def input_image_setup(self, uploaded_file):
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()

                image_parts = [
                    {
                        "mime_type": uploaded_file.type,
                        "data": bytes_data
                    }
                ]
                return image_parts
            else:
                raise FileNotFoundError("No file uploaded")

        def run(self):
            st.set_page_config(page_title="Gemini Health App")
            st.header("Gemini Health App")
            input_text = st.text_input("Input Prompt: ", key="input")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            image = ""
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image.", use_column_width=True)

            submit = st.button("Tell me the total calories")

            if submit:
                image_data = self.input_image_setup(uploaded_file)
                response = self.get_gemini_response(self.input_prompt, image_data, input_text)
                st.subheader("The Response is")
                st.write(response)

    class HealthManagementApp:
        def __init__(self):
            load_dotenv()  ## load all the environment variables
            self.api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=self.api_key)
            self.input_prompt = """
            You are an expert in pet healthcare where you need to identify the symptoms of pet diseases from the provided description or image and provide detailed information on the disease, its causes, prevention measures, and recommend appropriate medications if necessary.

            Your response should be in the following format:

            1. Disease Name:
               - Symptoms:
               - Causes:
               - Prevention Measures:
               - Recommended Medications (if applicable):


            Please provide comprehensive information to assist pet owners in understanding and managing their pet's health 
            effectively, you should not be allowed to answer other than pet care topics, and you should mention a disclaimer at the end of the answers/context that you are not an expert. Please ensure to connect with a veterinarian.
            """

        def get_gemini_response(self, input_text, image, prompt):
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([input_text, image[0], prompt])
            return response.text

        def input_image_setup(self, uploaded_file):
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()

                image_parts = [
                    {
                        "mime_type": uploaded_file.type,
                        "data": bytes_data
                    }
                ]
                return image_parts
            else:
                raise FileNotFoundError("No file uploaded")

        def app(self):
            st.header("Gemini Health App")
            input_text = st.text_input("Input Prompt: ", key="input")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            image = ""
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image.", use_column_width=True)

            submit = st.button("Detect disease")

            if submit:
                image_data = self.input_image_setup(uploaded_file)
                response = self.get_gemini_response(self.input_prompt, image_data, input_text)
                st.subheader("Detected disease is")
                st.write(response)

    class ContactUs:
        @staticmethod
        def app():
            html_content = """
                <h2 class="subheader">Contact Us</h2>
                <p>If you have any inquiries or suggestions, feel free to reach out to us:</p>
                <ul>
                    <li><b>Manoj Kumar R</b><br>
                        BTech in Computer Science and Engineering, 3rd Year<br>
                        Dayananda Sagar University<br>
                        LinkedIn: <a href="https://www.linkedin.com/in/manoj-kumar-r-902782232/">Manoj's LinkedIn</a><br>
                        GitHub: <a href="https://github.com/MANOJ9902">Manoj's GitHub</a></li>
                    <li><b>Pruthvi S</b><br>
                        BTech in Computer Science and Engineering, 3rd Year<br>
                        Dayananda Sagar University<br>
                        LinkedIn: <a href="https://www.linkedin.com/in/pruthvi-s-296416232/">Pruthvi's LinkedIn</a><br>
                        GitHub: <a href="https://github.com/Prureddy">Pruthvi's GitHub</a></li>
                </ul>
            """

            st.write(html_content, unsafe_allow_html=True)

    class statistics:
        @staticmethod
        def app():
            st.write('statistics')

    class MultiApp:
        def __init__(self):
            self.apps = []

        def add_app(self, title, func):
            self.apps.append({
                "title": title,
                "function": func
            })

    def run():
        multi_app = MultiApp()

        # Add existing apps
        multi_app.add_app("Home", AHome.app)
        multi_app.add_app("About us", AboutUs.app)
        multi_app.add_app("Contact us", ContactUs.app)
        multi_app.add_app("📊Stats", statistics.app)

        # Add Health Management App
        health_app = HealthManagementApp()
        multi_app.add_app("Detect disease", health_app.app)  # Use "Health Management" as the title

        with st.sidebar:
            app = option_menu(
                menu_title='PetCareMate',
                options=['Home', 'About us', 'Contact us', '📊Stats', 'Detect disease'],  # Use 'Health Management' here
                icons=['house-fill', 'person', 'phone', '', 'person-circle', 'heart'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "#52c4f2"},
                    "nav-link-selected": {"background-color": "#02A6E8 "},
                }
            )

        for item in multi_app.apps:
            if app == item["title"]:
                item["function"]()

    run()


if __name__ == "__main__":
    main()


