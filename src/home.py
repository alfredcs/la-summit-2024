import streamlit as st
import sys
import os

module_paths = ["./", "./configs"]
file_path = "./data/"

for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))

from utility import *

st.set_page_config(page_title="MM-RAG Demo",page_icon="🩺",layout="wide")
st.title("Multimodal RAG Demo")

aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")


#@st.cache_data
#@st.cache_resource(TTL=300)
with st.sidebar:
    #----- RAG  ------ 
    st.header(':green[Multimodal RAG] :file_folder:')
    rag_update = rag_retrieval = False
    rag_on = st.select_slider(
        'Activate RAG',
        value='None',
        options=['None', 'Update', 'Retrieval'])
    if 'Update' in rag_on:
        upload_docs = st.file_uploader("Upload your doc here", accept_multiple_files=True, type=['pdf', 'doc', 'jpg', 'png'])
        # Amazon Bedrock KB only supports titan-embed-text-v1 not g1-text-02
        embedding_model_id = st.selectbox('Choose Embedding Model',('amazon.amazon.titan-embed-text-v1', 'amazon.titan-embed-image-v1'))
        rag_update = True
    elif 'Retrieval' in rag_on:
        rag_retrieval = True
                
    #----- Choose models  ------ 
    st.divider()
    st.title(':orange[Multimodal Config] :pencil2:') 
    option = st.selectbox('Choose Model',('anthropic.claude-3-haiku-20240307-v1:0', 
                                          'anthropic.claude-3-sonnet-20240229-v1:0', 
                                          'anthropic.claude-instant-v1',
                                          'anthropic.claude-v2:1'))


    st.write("------- Default parameters ----------")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_token = st.number_input("Maximum Output Token", min_value=0, value=1024, step=64)
    top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)
    top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=40)
    #candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
    stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation", value="\n\nHuman")

        
    # ---- Clear chat history ----
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        voice_prompt = ""
        #del st.session_state[record_audio]


###
# Streamlist Body
###
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#
if rag_update:
    # Update AOSS
    if upload_docs:
        empty_directory(file_path)
        upload_doc_names = [file.name for file in upload_docs]
        for upload_doc in upload_docs:
            bytes_data = upload_doc.read()
            with open(file_path+upload_doc.name, 'wb') as f:
                f.write(bytes_data)
        stats, status, kb_id = bedrock_kb_injection(file_path)
        msg = f'Total {stats}  to Amazon Bedrock Knowledge Base: {kb_id}  with status: {status}.'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎙️').write(msg)

elif rag_retrieval:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        #Do retrieval
        #msg = bedrock_kb_retrieval(prompt, option)
        msg = bedrock_kb_retrieval_advanced(prompt, option, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        
else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
        