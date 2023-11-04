import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.chains.question_answering import load_qa_chain
from supabase.client import create_client
import streamlit as st
from streamlit_chat import message
load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_cliente = create_client(supabase_url,supabase_service_key)
embeddings = OpenAIEmbeddings()
db = SupabaseVectorStore(client=supabase_cliente, embedding=embeddings, table_name='documents')
llm = ChatOpenAI(model='gpt-3.5-turbo-16k')
chain = load_qa_chain(llm, chain_type='stuff')

def consulta(query):
    docs = db.similarity_search(query)
    respuesta = chain.run(input_documents=docs, question=query)
    return respuesta

if 'historial' not in st.session_state:
    st.session_state['historial'] = []
if 'bot' not in st.session_state:
    st.session_state['bot'] = []
if 'usuario' not in st.session_state:
    st.session_state['usuario'] = []

contenedorPpal = st.container()
contenedorChat = st.container()

with contenedorPpal:
    st.title('Microbosque CECAR')
    st.write('---')

    with st.form(key='formulario', clear_on_submit=True):
        input_usaurio = st.text_input('Pregunta',placeholder='Escribe tu pregunta aquí')
        boton_submit = st.form_submit_button(label='Consultar')

        if input_usaurio and boton_submit:
            respuestaBot = consulta(input_usaurio)
            st.session_state['usuario'].append(input_usaurio)
            st.session_state['bot'].append(respuestaBot)

        if st.session_state['bot']:
            with contenedorChat:
                for i in reversed(range(len(st.session_state['bot']))):
                    message(st.session_state['usuario'][i], is_user=True, key=str(i)+'_usuario', avatar_style='fun-emoji')
                    message(st.session_state['bot'][i], key=str(i), avatar_style='icons')