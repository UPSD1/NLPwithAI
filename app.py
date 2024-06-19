from edIntegration import authenticate, ed, post_comment
from thirdPartyIntegration import update_neo4j_vectordb, load_llm, load_embedding_model
from search import vector_search, generate_response, configure_llm_only_chain, configure_qa_structure_rag_chain
from langchain_community.graphs import Neo4jGraph
import time

#Environmental Variable
import os
import base64
import requests
import re
from dotenv import load_dotenv

load_dotenv()
OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')
NEO4J_KEY = os.getenv('NEO4J_KEY')
CLAUDE_KEY = os.getenv('CLAUDE_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = "neo4j"

llm_name = "claude"
course_id = 58877

def extract_possible_image_url(content):
  """
  This function attempts to extract the URL from an image tag within the provided content (HTML string).

  Args:
      content: A string containing HTML content.

  Returns:
      The extracted URL (if found), or None if no image tag is found.

  Notes:
      This function only extracts the URL and doesn't guarantee it's a valid image.
  """
  
  # Search for the image tag using regular expressions
  match = re.search(r'<figure><image src="([^"]+)"', content)
  if match:
    # Extract the URL from the matched group (index 1)
    url = match.group(1)
    
    # Check for common image extensions (optional, for reference)
    # if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
    #   return url
    
    # You can add additional checks based on context (e.g., keywords) here
    
    return url
  
  # No image tag found
  return None

def main():
    status, _= authenticate(verbose=False)
    if status: #if authentication was successful
        print("logged in")

    #load model
    embeddings, _ = load_embedding_model()

    llm = load_llm(llm_name)

    # rag_chain: KG augmented response
    model = configure_qa_structure_rag_chain(
        llm, embeddings, embeddings_store_url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
    )

    # #get already existing db
    critique_vectordb = update_neo4j_vectordb(mode=1, index_name = "critique", node_label = "critique-instruction")
    exam_vectordb = update_neo4j_vectordb(mode=1, index_name = "exam_general", node_label = "exam-instruction")    


    # response = generate_response(critique_vectordb, "Give me steps on how to critique")['answer']
    # print(response)

    # #Semantic search
    # query = "What's the Ultimate Goal?"
    # result = vector_search(vectordb, query)
    # print(result)


    # #query the KG and Vector DB on just 1 thread 
    # thread_no = 952904 #5021784 #4974136 #5021770
    # details = ed.get_thread(thread_no)
    # #extract url and preprocess
    # url = extract_possible_image_url(details['content'])
    # data = requests.get(url).content
    # img_base64 = base64.b64encode(data).decode("utf-8")

    # rag_chain_img = configure_qa_structure_rag_chain(
    #     llm, embeddings, embeddings_store_url=NEO4J_URI, 
    #     username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
    #     img_base64=img_base64
    # )
    
    # print("trying to answer the question")
    # print(f"Question:{details['document']}")
    # result = rag_chain_img.invoke({"question": details['document']})["answer"]
    # print(result)
    # print("Answered")

    # #critiques
    # if details['category'].lower() == "critiques":
    #     print("Category --> Critiques")
    #     response = generate_response(critique_vectordb, details['document'])['answer']
    #     ed.post_comment(thread_no, response)
    #     print("Answered")
    #     return
    # #exam
    # if details['category'].lower() == "exams":
    #     print("Category --> Exams")
    #     response = generate_response(exam_vectordb, details['document'])['answer']
    #     ed.post_comment(thread_no, response)
    #     print("Answered")
    #     return
    
    # result = model.invoke({"question": details['document']})["answer"]
    # ed.post_comment(thread_no, result)
    # print("Answered")




    
    # result = model.invoke({"question": details['document']})["answer"]
    # print(f"\nquery > {details['document']}")
    # print(f"AI > {result}")
    # ed.post_comment(4947717, result)
    
    # while(True):
    #     print("Enter your query below")
    #     user_input = input()
    #     if (user_input.lower().strip() == 'exit') or (user_input.lower().strip() == 'quit'):
    #         break
    #     start = time.time()
    #     result = model.invoke({"question": user_input})["answer"]
    #     print(f"\nquery > {user_input}")
    #     print(f"AI > {result}")
    #     print()
    #     print(f"{time.time() - start: 0.4f}secs used\n")

    offset = 0
    thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )

    def answer_all_thread(thread_lst, offset):
        for i, thread in enumerate(thread_lst):
            print(f"Thread {offset+i+1} --> {thread['category']}")

            #if post is an announcement skip
            if thread["type"].lower() == "announcement":
                continue

            #Quizzes or Projects
            if thread['category'].lower() in ["projects","quizzes"]:
                time.sleep(1)
                #Recursively loop through all threads
                if i == len(thread_lst)-1:
                    offset += len(thread_lst)
                    thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )
                    answer_all_thread(thread_lst, offset)
                continue

            # get url if available
            url = extract_possible_image_url(thread['content'])
            # print(url)

            if url == None:
                #critiques
                if thread['category'].lower() == "critiques":
                    response = generate_response(critique_vectordb, thread['document'])['answer']
                    post_comment(thread['id'], response)
                    print("Answered...")

                    #Recursively loop through all threads
                    if i == len(thread_lst)-1:
                        offset += len(thread_lst)
                        thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )
                        answer_all_thread(thread_lst, offset)
                    continue
                #exam
                if thread['category'].lower() == "exams":
                    response = generate_response(exam_vectordb, thread['document'])['answer']
                    post_comment(thread['id'], response)
                    print("Answered...")

                    #Recursively loop through all threads
                    if i == len(thread_lst)-1:
                        offset += len(thread_lst)
                        thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )
                        answer_all_thread(thread_lst, offset)
                    continue
                
                result = model.invoke({"question": thread['document']})["answer"]
                # print(result)
                post_comment(thread['id'], result)
                print("Answered...")

            else:
                img = requests.get(url)
                data = img.content
                img_type = img.headers['Content-Type']
                img_base64 = base64.b64encode(data).decode("utf-8")
                # print(img_base64)

                #critiques
                if thread['category'].lower() == "critiques":
                    response = generate_response(critique_vectordb, thread['document'], img_base64 = img_base64, img_type = img_type)['answer']
                    post_comment(thread['id'], response)
                    print("Answered...")

                    #Recursively loop through all threads
                    if i == len(thread_lst)-1:
                        offset += len(thread_lst)
                        thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )
                        answer_all_thread(thread_lst, offset)
                    continue

                #exam
                if thread['category'].lower() == "exams":
                    response = generate_response(exam_vectordb, thread['document'],img_base64 = img_base64, img_type = img_type)['answer']
                    post_comment(thread['id'], response)
                    print("Answered...")

                    #Recursively loop through all threads
                    if i == len(thread_lst)-1:
                        offset += len(thread_lst)
                        thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )
                        answer_all_thread(thread_lst, offset)
                    continue
               
                rag_chain_img = configure_qa_structure_rag_chain(
                    llm, embeddings, embeddings_store_url=NEO4J_URI,
                    username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                    img_base64=img_base64, img_type = img_type
                    )
                result = rag_chain_img.invoke({"question": thread['document']})["answer"]
                post_comment(thread['id'], result)
                print("Answered...")
        
            #Recursively loop through all threads
            if i == len(thread_lst)-1:
                offset += len(thread_lst)
                thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" )
                answer_all_thread(thread_lst, offset)

            # break
        print("done")
    
    answer_all_thread(thread_lst, offset)
    print("Question Done")


    # counter = 0
    # latest_id = 5021430
    # while(True):
    #     thread_lst = ed.list_threads(course_id, limit = 30, offset = 0, sort = "new" )
    #     # counter += 1
    #     # print(counter)
    #     if(thread_lst[0]['id'] != latest_id):
    #         print("found new question")
    #         details = ed.get_thread(thread_lst[0]['id'])
    #         #get url if available
    #         url = extract_possible_image_url(details['content'])
    #         # print(url)

    #         if url == None:
    #             #critiques
    #             if details['category'].lower() == "critiques":
    #                 print("Category --> Critiques")
    #                 response = generate_response(critique_vectordb, details['document'])['answer']
    #                 print(response)
    #                 post_comment(thread_lst[0]['id'], response)
    #                 latest_id = thread_lst[0]['id']
    #                 print("Answered latest question")
    #                 time.sleep(1)
    #                 continue
    #             #exam
    #             if details['category'].lower() == "exams":
    #                 print("Category --> Exams")
    #                 response = generate_response(exam_vectordb, details['document'])['answer']
    #                 print(response)
    #                 post_comment(thread_lst[0]['id'], response)
    #                 latest_id = thread_lst[0]['id']
    #                 print("Answered latest question")
    #                 time.sleep(1)
    #                 continue
                
    #             result = model.invoke({"question": details['document']})["answer"]
    #             post_comment(thread_lst[0]['id'], result)
    #             latest_id = thread_lst[0]['id']
    #             print("Answered latest question")
    #         else:
    #             data = requests.get(url).content
    #             img_base64 = base64.b64encode(data).decode("utf-8")

    #             #critiques
    #             if details['category'].lower() == "critiques":
    #                 print("Category --> Critiques")
    #                 response = generate_response(critique_vectordb, details['document'], img_base64)['answer']
    #                 print(response)
    #                 post_comment(thread_lst[0]['id'], response)
    #                 latest_id = thread_lst[0]['id']
    #                 print("Answered latest question")
    #                 time.sleep(1)
    #                 continue
    #             #exam
    #             if details['category'].lower() == "exams":
    #                 print("Category --> Exams")
    #                 response = generate_response(exam_vectordb, details['document'], img_base64)['answer']
    #                 print(response)
    #                 post_comment(thread_lst[0]['id'], response)
    #                 latest_id = thread_lst[0]['id']
    #                 print("Answered latest question")
    #                 time.sleep(1)
    #                 continue

    #             rag_chain_img = configure_qa_structure_rag_chain(
    #                 llm, embeddings, embeddings_store_url=NEO4J_URI,
    #                 username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
    #                 img_base64=img_base64
    #                 )
    #             result = rag_chain_img.invoke({"question": details['document']})["answer"]
    #             post_comment(thread_lst[0]['id'], result)
    #             latest_id = thread_lst[0]['id']
    #             print("Answered latest question")
                       
    #     time.sleep(1)
        



if __name__ == "__main__":
    main()