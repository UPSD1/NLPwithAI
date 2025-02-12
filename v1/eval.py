from edapi import EdAPI
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import time

load_dotenv()
OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPEN_AI_KEY)
course_id_original = 12801
course_id_duplicate = 58877
priority_ids = {146168, 336652, 133439, 304091, 155425, 151439, 147345, 132907, 132929}
skip_number = []
skipThreadNo = []
# initialize Ed API
ed = EdAPI()

# authenticate user through the ED_API_TOKEN environment variable
ed.login()

# retrieve user information; authentication is persisted to next API calls
user_info = ed.get_user_info()
user = user_info['user']
print(f"Hello {user['name']}!")

# def get_average_and_replace(data_dict):
#   """
#   This function takes a dictionary where the values are lists and replaces
#   each list with the average of its elements.

#   Args:
#       data_dict: A dictionary where the keys are any data type and the values are lists of numbers.

#   Returns:
#       The modified dictionary with the averages replacing the original lists.
#   """

#   for key, value_list in data_dict.items():
#     # Check if the value is a list of numbers
#     if all(isinstance(x, (int, float)) for x in value_list):
#       # Calculate the average of the list elements
#       average = sum(value_list) / len(value_list)
#       # Replace the list with the average in the dictionary
#       data_dict[key] = average
  
#   return data_dict
def get_average_and_replace(data_dict):
  for key, value_list in data_dict.items():
    if all(isinstance(x, (int, float)) for x in value_list):
      data_dict[key] = sum(value_list) / len(value_list)
  return data_dict

# def cosine_similarity(embedding1, embedding2):
#   """
#   Calculates the cosine similarity between two embedding vectors.

#   Args:
#       embedding1: A numpy array representing the first embedding vector.
#       embedding2: A numpy array representing the second embedding vector.

#   Returns:
#       A float value between -1 and 1 representing the cosine similarity.
#   """
#   # Ensure both embeddings have the same size
#   if len(embedding1) != len(embedding2):
#     raise ValueError("Embeddings must have the same size")

#   # Calculate the dot product
#   dot_product = np.dot(embedding1, embedding2)

#   # Calculate the magnitudes of the vectors
#   magnitude1 = np.linalg.norm(embedding1)
#   magnitude2 = np.linalg.norm(embedding2)

#   # Prevent division by zero
#   if magnitude1 == 0 or magnitude2 == 0:
#     return 0

#   # Calculate the cosine similarity
#   cosine_similarity = dot_product / (magnitude1 * magnitude2)
#   return cosine_similarity
def cosine_similarity(embedding1, embedding2):
  dot_product = np.dot(embedding1, embedding2)
  magnitude1 = np.linalg.norm(embedding1)
  magnitude2 = np.linalg.norm(embedding2)
  if magnitude1 == 0 or magnitude2 == 0:
    return 0
  return dot_product / (magnitude1 * magnitude2)

def find_similar_title(title, content, offset, thread_no):
    thread_lst_original = ed.list_threads(course_id_original, limit = 100, offset = offset, sort = "new" )
    
    #if we have no thread again
    if not thread_lst_original:
      print("Ran through all threads and couldn't find a match")
      skip_number.append(thread_no)
      print(f"Threads to checkout{skip_number}")
      return
    
    for i , orig_thread in enumerate(thread_lst_original):
      if (orig_thread["type"].lower() == "announcement") or (orig_thread['category'].lower() in ["projects","quizzes"]):
        if i == (len(thread_lst_original) - 1):
          result = find_similar_title(title, content, offset=offset + len(thread_lst_original), thread_no=thread_no)
          return result
        continue

      #get the id
      orig_thread_id = orig_thread['id']
      orig_title = orig_thread['title']
      orig_content = orig_thread['document']
      
      #if the two title are the same and the questions also match
      if (orig_title == title) and (orig_content == content):
          print(f"thread id {orig_thread_id}")
          #get the answer attached the thread by TA
          data = ed.get_thread(orig_thread_id)

          if data['answers'] == []:
             print("Not answered in main")
             return
          #check if we have multiple responses
          if len(data['answers']) >= 1:
              for answer in data['answers']:
                  if answer['user_id'] in priority_ids:
                      result = answer['document']
                      return result    
              print("Question not answered by a staff")
              return
      
      if i == (len(thread_lst_original) - 1):
        result = find_similar_title(title, content,offset=offset + len(thread_lst_original), thread_no=thread_no)
        return result

#retrieve list thread in a course
offset = 0
thread_lst_duplicate = ed.list_threads(course_id_duplicate, limit = 100, offset = offset, sort = "new" )

evaluation_scores = []
evaluation_dict = {}
count = 0 #used to find how many questions appeared in both  courses

def evaluate(thread_lst_duplicate, offset):
  if not thread_lst_duplicate:
    print("end of thread")
    return
  
  #loop through
  for i, thread in enumerate(thread_lst_duplicate):
    print(f"Thread {offset+i+1}")

    if (thread["type"].lower() == "announcement") or (thread['category'].lower() in ["projects","quizzes"]):
        if i == (len(thread_lst_duplicate) - 1):
          offset = offset + len(thread_lst_duplicate)
          thread_lst_duplicate = ed.list_threads(course_id_duplicate, limit = 100, offset = offset, sort = "new" )
          evaluate(thread_lst_duplicate, offset=offset)
        continue
    
    #get the id
    thread_id = thread['id']
    title = thread['title']
    content = thread['document']

    thread_data = ed.get_thread(thread_id)
    ai_resp = thread_data['answers'][-1]['document'] if thread_data['answers'] else None
    # ai_resp = ed.get_thread(thread_id)['answers'][0]['document'] if ed.get_thread(thread_id)['answers'] else None
    #if the AI answered
    if ai_resp:
      ta_resp = find_similar_title(title=title, content = content, offset=0, thread_no=offset+i)
    else:
      print(f"AI didn't answer this question thread number -->{thread_id}")
      continue

    if (ta_resp == None):
      print(f"Couldn't find a match; thread number --> {thread_id}")
      if offset+i in skip_number:
         skipThreadNo.append(thread_id)
         print(f"Skipped thread Nos {skipThreadNo}")
      if i == (len(thread_lst_duplicate) - 1):
        offset = offset + len(thread_lst_duplicate)
        thread_lst_duplicate = ed.list_threads(course_id_duplicate, limit = 100, offset = offset, sort = "new" )
        evaluate(thread_lst_duplicate, offset=offset)
      continue
    
    print(title)

    embedding1 = embeddings.embed_query(ta_resp)
    embedding2 = embeddings.embed_query(ai_resp)

    similarity = cosine_similarity(embedding1, embedding2)
    print("Cosine similarity:", similarity)

    category = thread['category']
    print(category)
    try:
        evaluation_dict[category].append(similarity)
    except:
        evaluation_dict[category] = [similarity]

    evaluation_scores.append(similarity)
    print(evaluation_dict)
    print("\n\n")

evaluate(thread_lst_duplicate, offset)

print(evaluation_scores)
print(f"Overall evaluation is {sum(evaluation_scores)/len(evaluation_scores)}")

print(f"Evaluation dict ---> {evaluation_dict}")
print(f"Final result: \n{get_average_and_replace(evaluation_dict)}")



# {'Lectures': 0.5019341829519125, 'General': 0.4349365773674671, 'Exams': 0.40371338034625753, 'Critiques': 0.5141907423480416}
