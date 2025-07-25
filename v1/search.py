#----------------------------------Import Libraries-----------------------------------------
#langchain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage

#others
from typing import List, Any #List type hint from the typing module
from thirdPartyIntegration import claude_model, openai_model

memory = ConversationBufferMemory(memory_key="chat_history", 
                                  input_key="question", 
                                  output_key="answer",
                                  return_messages=True,
                                  k = 5)
#-------------------------------------------------------------------------------------------

def vector_search(vectordb, query,db = "neo4j"):
    if db.lower() == "neo4j":
        results = vectordb.similarity_search(query, k=4)
        return results
    elif db.lower() == "pinecone":
        results = vectordb.similarity_search(query, k=4)
        return results

def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps with answering general questions.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        answer = llm(
            prompt.format_prompt(
                text=user_input,
            ).to_messages(),
            callbacks=callbacks,
        ).content
        return {"answer": answer}

    return generate_llm_output

def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response
    general_system_template = """
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs and their links from Stackoverflow.
    You should prefer information from accepted or more upvoted answers.
    Make sure to rely on information from the answers and not on questions to provide accuate responses.
    When you find particular answer in the context useful, make sure to cite it in the answer using the link.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    Each answer you generate should contain a section at the end of links to
    Stackoverflow questions and answers you found useful, which are described under Source value.
    You can only use links to StackOverflow questions that are present in the context and always
    add links to the end of the answer in the style of citations.
    Generate concise answers with references sources section of links to
    relevant StackOverflow questions only at the end of the answer.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="stackoverflow",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query="""
    WITH node AS question, score AS similarity
    CALL  { with question
        MATCH (question)<-[:ANSWERS]-(answer)
        WITH answer
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH collect(answer)[..2] as answers
        RETURN reduce(str='', answer IN answers | str +
                '\n### Answer (Accepted: '+ answer.is_accepted +
                ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    }
    RETURN '##Question: ' + question.title + '\n' + question.body + '\n'
        + answerTexts AS text, similarity as score, {source: question.link} AS metadata
    ORDER BY similarity ASC // so that best answers are the last
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kg_qa

def configure_qa_structure_rag_chain(llm, embeddings, embeddings_store_url, username, password, img_base64 = None, img_type = None):
    # RAG response based on vector search and retrieval of structured chunks

    sample_query = """
    // 0 - prepare question and its embedding
        MATCH (ch:Chunk) -[:HAS_EMBEDDING]-> (chemb)
        WHERE ch.block_idx = 19
        WITH ch.sentences AS question, chemb.value AS qemb
        // 1 - search chunk vectors
        CALL db.index.vector.queryNodes($index_name, $k, qemb) YIELD node, score
        // 2 - retrieve connectd chunks, sections and documents
        WITH node AS answerEmb, score
        MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
        WITH s, score LIMIT 1
        MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
        WITH d, s, chunk, score ORDER BY chunk.block_idx ASC
        // 3 - prepare results
        WITH d, collect(chunk) AS chunks, score
        RETURN {source: d.url, page: chunks[0].page_idx} AS metadata,
            reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, score;
    """

    # general_system_template = """
    # As a Teaching Assistant, your role is to guide students towards the correct answer while providing them with opportunities to learn and grow. 
    # When a student asks a question, follow these steps:
    # 1. Acknowledge the student's question and provide positive reinforcement for their effort.
    # 2. Assess the student's current understanding of the topic based on their question.
    # 3. Break down the problem or concept into smaller, more manageable parts.
    # 4. Provide hints, examples, or analogies to help the student understand the underlying principles.
    # 5. Encourage the student to think critically and arrive at the answer independently.
    # 6. If the student struggles, offer more specific guidance while still allowing them to take an active role in the learning process.
    # 7. Once the student arrives at the correct answer, provide feedback and reinforce their understanding.
    # 8. Offer additional resources or practice problems to help the student solidify their knowledge.
    
    # Remember, your goal is not to simply give away the answer but to empower the student to develop their problem-solving skills and deep understanding of the subject matter. 
    # Maintain a supportive and patient demeanor throughout the interaction.
    
    # Use the following context to answer the question at the end.
    # Make sure not to make any changes to the context if possible when prepare answers so as to provide accuate responses.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # ----
    # {summaries}
    # ----
    # At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    # For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    # """
    general_system_template = """
    As a teaching assistant for univeristy students assist students to understand better and get the right answer
    Use the following context to assist in answering the question at the end.
    Make sure not to make any changes to the context if possible when preparing answers so as to provide accuate responses.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    """
    general_user_template = "Question:```{question}```"
    general_user_template_with_img = [
        {
            "type": "image_url",
            "image_url": {
                # langchain logo
                "url": f"data:{img_type};base64,{img_base64}",  # noqa: E501
            },
        },
        {"type": "text", "text": general_user_template},
        ]
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    messages_with_image = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessage(content=general_user_template_with_img)
    ]

    if img_base64:
        print("using image prompt")
        qa_prompt = ChatPromptTemplate.from_messages(messages_with_image)
    else:
        qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="chunkVectorIndex",  # vector by default
        node_label="Embedding",  # embedding node label
        embedding_node_property="value",  # embedding value property
        text_node_property="sentences",  # text by default
        retrieval_query="""
            WITH node AS answerEmb, score
            ORDER BY score DESC LIMIT 10
            MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
            WITH s, answer, score
            MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
            WITH d, s, answer, chunk, score ORDER BY d.url_hash, s.title, chunk.block_idx ASC
            // 3 - prepare results
            WITH d, s, collect(answer) AS answers, collect(chunk) AS chunks, max(score) AS maxScore
            RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata,
                reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, maxScore AS score LIMIT 3;
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 25}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=7000,      # gpt-4
        memory=memory,
    )

    return kg_qa

def generate_response(vectordb, query,model_name = "finetuned", img_base64 = None, img_type = None):
    if model_name.lower() == "anthropic":
        model = claude_model
    else:
        model = openai_model

    # general_system_template = """
    # As a Teaching Assistant, your role is to guide students towards the correct answer while providing them with opportunities to learn and grow. 
    # When a student asks a question, follow these steps:
    # 1. Acknowledge the student's question and provide positive reinforcement for their effort.
    # 2. Assess the student's current understanding of the topic based on their question.
    # 3. Break down the problem or concept into smaller, more manageable parts.
    # 4. Provide hints, examples, or analogies to help the student understand the underlying principles.
    # 5. Encourage the student to think critically and arrive at the answer independently.
    # 6. If the student struggles, offer more specific guidance while still allowing them to take an active role in the learning process.
    # 7. Once the student arrives at the correct answer, provide feedback and reinforce their understanding.
    # 8. Offer additional resources or practice problems to help the student solidify their knowledge.
    
    # Remember, your goal is not to simply give away the answer but to empower the student to develop their problem-solving skills and deep understanding of the subject matter. 
    # Maintain a supportive and patient demeanor throughout the interaction.
    
    # Use the following context to answer the question at the end.
    # Make sure not to make any changes to the context if possible when prepare answers so as to provide accuate responses.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # ----
    # {summaries}
    # ----
    # At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    # For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    # """
    general_system_template = """
    As a teaching assistant for univeristy students assist students to understand better and get the right answer
    Use the following context to assist in answering the question at the end.
    Make sure not to make any changes to the context if possible when preparing answers so as to provide accuate responses.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    """
    general_user_template = "Question:```{question}```"
    general_user_template_with_img = [
        {
            "type": "image_url",
            "image_url": {
                # langchain logo
                "url": f"data:{img_type};base64,{img_base64}",  # noqa: E501
            },
        },
        {"type": "text", "text": general_user_template},
        ]
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    messages_with_image = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessage(content=general_user_template_with_img)
    ]

    if img_base64:
        print("using image prompt")
        qa_prompt = ChatPromptTemplate.from_messages(messages_with_image)
    else:
        qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        model,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=vectordb.as_retriever(),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=7000,
        memory=memory,
    )

    result = chain.invoke({"question": query},return_only_outputs=True)

    return result