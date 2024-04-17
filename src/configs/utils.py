import arxiv
from langchain_core.documents.base import Document
import requests
import os
import boto3
import multiprocessing
from bs4 import BeautifulSoup
import requests # Required to make HTTP requests
from bs4 import BeautifulSoup  # Required to parse HTML
from urllib.parse import unquote # Required to unquote URLs
from xml.etree import ElementTree

from operator import itemgetter
from langchain import hub
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
#from langchain.utilities import GoogleSearchAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from readabilipy import simple_json_from_html_string # Required to parse HTML to pure text

from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
# To address  RuntimeError: maximum recursion depth exceeded
import sys   
sys.setrecursionlimit(10000)

## SEarch engines ----

#----------- Parse out web content -----------
def scrape_and_parse(url: str) -> Document:
    """Scrape a webpage and parse it into a Document object"""
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=True)
    # The following line seems to work with the package versions on my local machine, but not on Google Colab
    # return Document(page_content=article['plain_text'][0]['text'], metadata={'source': url, 'page_title': article['title']})
    return Document(page_content='\n\n'.join([a['text'] for a in article['plain_text']]), metadata={'source': url, 'page_title': article['title']})


# Saerch google and bing with a query and return urls
def google_search(query: str, num_results: int=5):
    google_search = GoogleSearchAPIWrapper(google_api_key=os.getenv("google_api_key"), google_cse_id=os.getenv("google_cse_id"))
    google_results = google_search.results(query, num_results=num_results)
    documents = []
    urls = []
    for item in google_results:
        try:
            # Send a GET request to the URL
            response = requests.get(item['link'])
            print(f"Before:{item['link']}")
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract the text content from the HTML
            content = soup.get_text()
            if "404 Not Found" not in content:
                # Create a LangChain document
                doc = Document(page_content=content, metadata={'title': item['title'],'source': item['link']})
                print(f"After: {item['link']}")
                documents.append(doc)
                urls. append(item['link'])
    
        except requests.exceptions.RequestException as e:
            print(f"Error parsing URL: {e}")
            pass

    return documents, urls

'''
class newsSearcher:
    def __init__(self):
        self.google_url = "https://www.google.com/search?q="
        self.bing_url = "https://www.bing.com/search?q="
        #self.bing_url = "https://www.bing.com/search?q={query.replace(' ', '+')}"

    def search(self, query, count: int=3):
        with multiprocessing.Pool(processes=2) as pool:
            google_results = pool.apply_async(self.search_google, args=(query,count))
            bing_results = pool.apply_async(self.search_bing, args=(query,count))
    
            # Get results from both processes
            google_urls = google_results.get()
            bing_urls = bing_results.get()
        #google_urls = self.search_goog(query, count)
        #bing_urls = self.search_bing(query, count)
        combined_urls = google_urls + bing_urls
        urls = list(set(combined_urls))  # Remove duplicates
        return [scrape_and_parse(f) for f in google_urls], urls # Scrape and parse all the url

    def search_goog(self, query, count):
        #response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML
        #soup = BeautifulSoup(response.text, "html.parser").get_text() # Parse the HTML
        links = soup.find_all("a", recursive=True) # Find all the links in the HTML
        urls = []
        for l in [link for link in links if link["href"].startswith("/url?q=")]:
            # get the url
            url = l["href"]
            # remove the "/url?q=" part
            url = url.replace("/url?q=", "")
            # remove the part after the "&sa=..."
            url = unquote(url.split("&sa=")[0])
            # special case for google scholar
            if url.startswith("https://scholar.google.com/scholar_url?url=http"):
                url = url.replace("https://scholar.google.com/scholar_url?url=", "").split("&")[0]
            elif 'google.com/' in url: # skip google links
                continue
            elif 'youtube.com/' in url:
                continue
            elif 'search?q=' in url:
                continue
            if url.endswith('.pdf'): # skip pdf links
                continue
            if '#' in url: # remove anchors (e.g. wikipedia.com/bob#history and wikipedia.com/bob#genetics are the same page)
                url = url.split('#')[0]
            # print the url
            urls.append(url)
        return urls
        
    def search_google(self, query, count):
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".yuRUbf a")]
        return urls

    def search_bing(self, query, count):
        params = {
            "q": query,
            "count": count # Number of results to retrieve
        }
        response = requests.get(self.bing_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".b_algo h2 a")]
        return urls[:count]
'''

# ---- Search arxiv -----
#--- Configure Bedrock -----
def config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k):
    bedrock_client = boto3.client('bedrock-runtime')
    embedding_bedrock = BedrockEmbeddings(client=bedrock_client, model_id=embedding_model_id)
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        #"stop_sequences": ["\n\nHuman"],
    }
    chat = BedrockChat(
        model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    )
    #llm = Bedrock(
    #    model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    #)

    return chat, embedding_bedrock
    
# Convert to Document
def convert_to_document2(entry):
    """Convert an entry to a LangChain Document object."""
    # Adjust attributes according to the actual Document class definition
    document = Document(
        page_content=entry.summary,
        metadata={
            'title': entry.title,
            'authors': entry.authors,
            'id': entry.entry_id,
            'link': entry.links,
            'categories':entry.categories,
            'published': entry.published
        }
    )
    return document


def parse_response(xml_data):
    """Parse the XML response from arXiv and extract relevant data."""
    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    root = ElementTree.fromstring(xml_data)
    entries_data = []
    
    for entry in root.findall('arxiv:entry', namespace):
        entry_data = {
            'title': entry.find('arxiv:title', namespace).text,
            'summary': entry.find('arxiv:summary', namespace).text,
            'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
            'id': entry.find('arxiv:id', namespace).text,
            'link': entry.find('arxiv:link[@rel="alternate"]', namespace).attrib.get('href'),
            'published': entry.find('arxiv:published', namespace).text
        }
        entries_data.append(entry_data)
    
    return entries_data

def download_pdf(entries, dest_filepath):
    for entry in entries:
        paper_id = entry['id'].split("/")[-1]
        file_name = f"{dest_filepath}/{paper_id}.pdf"
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        paper.download_pdf(filename=file_name)  # Downloads the paper
    return True

def download_pdf2(entries, dest_filepath):
    for entry in entries:
        paper_id = entry['id'].split("/")[-1]
        file_name = f"{dest_filepath}/{paper_id}.pdf"
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        paper.download_pdf(filename=file_name)  # Downloads the paper
    return True

def convert_to_document(entry):
    """Convert an entry to a LangChain Document object."""
    # Adjust attributes according to the actual Document class definition
    document = Document(
        page_content=entry['summary'],
        metadata={
            'title': entry['title'],
            'authors': entry['authors'],
            'id': entry['id'],
            'link': entry['link'],
            'published': entry['published']
        }
    )
    return document

def search_and_convert(query, max_results=10, filepath='pdfs'):
    """Search arXiv, parse the results, and convert them to LangChain Document objects."""
    params = {"search_query": query, "start": 0, "max_results": max_results}
    base_url = "http://export.arxiv.org/api/query?"
    os.makedirs(filepath, exist_ok=True)
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        entries = parse_response(response.text)
        download_pdf(entries,filepath) 
        return [convert_to_document(entry) for entry in entries]
    else:
        print(f"Error fetching results from arXiv: {response.status_code}")
        return []

# Construct the default API client
def search_arxiv(query:str, max_results=10, filepath: str='pdfs'):
    client = arxiv.Client()
    docs = []
    # Search for articles matching the keyword "question and answer"
    search = arxiv.Search(
      query = query,
      max_results = max_results,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )
    results = client.results(search)
    #download_pdf2(results, filepath)
    for result in results:
        docs.append(convert_to_document2(result))
    return docs

# ---- 
def retrieval_faiss(query, documents, model_id, embedding_model_id:str, chunk_size:int=6000, over_lap:int=600, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=25, doc_num: int=3):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap)
    docs = text_splitter.split_documents(documents)
    
    # Prepare embedding function
    chat, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    
    # Try to get vectordb with FAISS
    db = FAISS.from_documents(docs, embedding)
    retriever = db.as_retriever(search_kwargs={"k": doc_num})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    messages = [
        ("system", """Your are a helpful assistant to provide omprehensive and truthful answers to questions, \n
                    drawing upon all relevant information contained within the specified in {context}. \n 
                    You add value by analyzing the situation and offering insights to enrich your answer. \n
                    Simply say I don't know if you can not find any evidence to match the question. \n
                    """),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)

    # Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor= FlashrankRerank(), base_retriever=retriever
    )

    rag_chain = (
        #{"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        #| RunnableParallel(answer=hub.pull("rlm/rag-prompt") | chat |format_docs, question=itemgetter("question") ) 
        RunnableParallel(context=compression_retriever | format_docs, question=RunnablePassthrough() )
        | prompt_template
        | chat
        | StrOutputParser()
    )

    results = rag_chain.invoke(query)
    return results
