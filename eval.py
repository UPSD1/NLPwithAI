from edapi import EdAPI
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPEN_AI_KEY = os.getenv('OPEN_AI_SECRET_KEY')

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPEN_AI_KEY)
course_id_original = 12801
course_id_duplicate = 58877
priority_ids = [146168]
skip_number = [2, 5, 6, 7, 9, 11]
# initialize Ed API
ed = EdAPI()

# authenticate user through the ED_API_TOKEN environment variable
ed.login()

# retrieve user information; authentication is persisted to next API calls
user_info = ed.get_user_info()
user = user_info['user']
print(f"Hello {user['name']}!")

def cosine_similarity(embedding1, embedding2):
  """
  Calculates the cosine similarity between two embedding vectors.

  Args:
      embedding1: A numpy array representing the first embedding vector.
      embedding2: A numpy array representing the second embedding vector.

  Returns:
      A float value between -1 and 1 representing the cosine similarity.
  """
  # Ensure both embeddings have the same size
  if len(embedding1) != len(embedding2):
    raise ValueError("Embeddings must have the same size")

  # Calculate the dot product
  dot_product = np.dot(embedding1, embedding2)

  # Calculate the magnitudes of the vectors
  magnitude1 = np.linalg.norm(embedding1)
  magnitude2 = np.linalg.norm(embedding2)

  # Prevent division by zero
  if magnitude1 == 0 or magnitude2 == 0:
    return 0

  # Calculate the cosine similarity
  cosine_similarity = dot_product / (magnitude1 * magnitude2)
  return cosine_similarity

def find_similar_title(title, offset):
#    print("Entered")
   thread_lst_original = ed.list_threads(course_id_original, limit = 100, offset = offset, sort = "new" )
   
   #if we have no thread again
   if thread_lst_original == []:
      return
   
   for i , orig_thread in enumerate(thread_lst_original):
        # print(f"Thread number {offset + i}")
        #get the id
        orig_thread_id = orig_thread['id']

        #use the id to get the title
        orig_title = ed.get_thread(orig_thread_id)['title'].lower().strip()
        
        #if the two title are the same
        if orig_title == title:
            print(f"thread id {orig_thread_id}")
            #get the answer attached the thread by TA
            if len(ed.get_thread(orig_thread_id)['answers']) > 1:
               for answer in ed.get_thread(orig_thread_id)['answers']:
                  if answer['user_id'] in priority_ids:
                     result = answer['document']
                     return result
            
            response = ed.get_thread(orig_thread_id)['answers'][0]['document'] if ed.get_thread(orig_thread_id)['answers'] else None
            return response
        
        if i == (len(thread_lst_original) - 1):
            result = find_similar_title(title, offset=offset + len(thread_lst_original))
            return result

#retrieve list thread in a course
thread_lst_duplicate = ed.list_threads(course_id_duplicate, limit = 100, offset = 0, sort = "new" )

evaluation_scores = []
count = 0 #used to find how many questions appeared in both  courses

for i, thread in enumerate(thread_lst_duplicate):
    print(f"Thread {i} of {len(thread_lst_duplicate)}")
    if i in skip_number:
       continue
    #get the id
    thread_id = thread['id']
    title = ed.get_thread(thread_id)['title'].lower().strip()

    ta_resp = find_similar_title(title=title, offset=0)
    ai_resp = ed.get_thread(thread_id)['answers'][0]['document'] if ed.get_thread(thread_id)['answers'] else None

    if (ta_resp == None) or (ai_resp == None):
       continue
    
    count += 1
    print(count)
    print(title)

    embedding1 = embeddings.embed_query(ta_resp)
    embedding2 = embeddings.embed_query(ai_resp)

    similarity = cosine_similarity(embedding1, embedding2)
    print("Cosine similarity:", similarity)
    evaluation_scores.append(similarity)
    print("\n\n")

print(evaluation_scores)
print(f"Overall evaluation is {sum(evaluation_scores)/len(evaluation_scores)}")

# ss = [{'id': 4998433, 'user_id': 991487, 'course_id': 58877, 'original_id': None, 'editor_id': None, 'accepted_id': None, 'duplicate_id': None, 'number': 30, 'type': 'post', 'title': 'Overfitting with LR Model classifiers', 'content': '<document version="2.0"><paragraph>Hi,</paragraph><paragraph>Just wanted to know how LR classifiers deal with overfitting ( too many features and too few data points). We did have a gentle introduction to feature selection, but could you help share other techniques that could mitigate this commonly seen issue while training classifiers. ..?</paragraph></document>', 'document': 'Hi,\n\nJust wanted to know how LR classifiers deal with overfitting ( too many features and too few data points). We did have a gentle introduction to feature selection, but could you help share other techniques that could mitigate this commonly seen issue while training classifiers. ..?', 'category': 'Lectures', 'subcategory': '', 'subsubcategory': '', 'flag_count': 0, 'star_count': 0, 'view_count': 6, 'unique_view_count': 1, 'vote_count': 0, 'reply_count': 0, 'unresolved_count': 0, 'is_locked': False, 'is_pinned': False, 'is_private': False, 'is_endorsed': False, 'is_answered': True, 'is_student_answered': False, 'is_staff_answered': True, 'is_archived': False, 'is_anonymous': False, 'is_megathread': False, 'anonymous_comments': False, 'approved_status': 'approved', 'created_at': '2024-05-29T02:26:31.965554+10:00', 'updated_at': '2024-05-30T05:11:30.161909+10:00', 'deleted_at': None, 'pinned_at': None, 'anonymous_id': 0, 'vote': 0, 'is_seen': True, 'is_starred': False, 'is_watched': True, 'glanced_at': '2024-05-30T05:11:30.162689+10:00', 'new_reply_count': 0, 'duplicate_title': None, 'user': {'id': 991487, 'role': 'user', 'name': 'David Akinboro', 'avatar': None, 'course_role': 'admin', 'tutorials': {'58877': ''}}}, {'id': 4998408, 'user_id': 991487, 'course_id': 58877, 'original_id': None, 'editor_id': None, 'accepted_id': None, 'duplicate_id': None, 'number': 29, 'type': 'post', 'title': 'Chunking', 'content': '<document version="2.0"><paragraph>So is chunking like POS tagging but just at a more shallow level? Like now instead were just assigning labels to phrases instead of the individual tokens?</paragraph></document>', 'document': 'So is chunking like POS tagging but just at a more shallow level? Like now instead were just assigning labels to phrases instead of the individual tokens?', 'category': 'Lectures', 'subcategory': '', 'subsubcategory': '', 'flag_count': 0, 'star_count': 0, 'view_count': 9, 'unique_view_count': 1, 'vote_count': 0, 'reply_count': 0, 'unresolved_count': 0, 'is_locked': False, 'is_pinned': False, 'is_private': False, 'is_endorsed': False, 'is_answered': True, 'is_student_answered': False, 'is_staff_answered': True, 'is_archived': False, 'is_anonymous': True, 'is_megathread': False, 'anonymous_comments': False, 'approved_status': 'approved', 'created_at': '2024-05-29T02:13:27.144233+10:00', 'updated_at': '2024-05-30T05:11:29.548276+10:00', 'deleted_at': None, 'pinned_at': None, 'anonymous_id': 62690584, 'vote': 0, 'is_seen': True, 'is_starred': False, 'is_watched': True, 'glanced_at': '2024-05-30T05:11:29.549037+10:00', 'new_reply_count': 0, 'duplicate_title': None, 'user': {'id': 991487, 'role': 'user', 'name': 'David Akinboro', 'avatar': None, 'course_role': 'admin', 'tutorials': {'58877': ''}}}, {'id': 4998318, 'user_id': 991487, 'course_id': 58877, 'original_id': None, 'editor_id': None, 'accepted_id': None, 'duplicate_id': None, 'number': 28, 'type': 'post', 'title': 'word2vec', 'content': '<document version="2.0"><paragraph>Hi,</paragraph><paragraph>May I know how word2vec embedding vectors, when compared to one hot vectors, helps generalize better to rare words but not to unseen words ?</paragraph></document>', 'document': 'Hi,\n\nMay I know how word2vec embedding vectors, when compared to one hot vectors, helps generalize better to rare words but not to unseen words ?', 'category': 'Lectures', 'subcategory': '', 'subsubcategory': '', 'flag_count': 0, 'star_count': 0, 'view_count': 13, 'unique_view_count': 1, 'vote_count': 0, 'reply_count': 0, 'unresolved_count': 0, 'is_locked': False, 'is_pinned': False, 'is_private': False, 'is_endorsed': False, 'is_answered': True, 'is_student_answered': False, 'is_staff_answered': True, 'is_archived': False, 'is_anonymous': False, 'is_megathread': False, 'anonymous_comments': False, 'approved_status': 'approved', 'created_at': '2024-05-29T01:28:25.308185+10:00', 'updated_at': '2024-05-30T05:11:28.991489+10:00', 'deleted_at': None, 'pinned_at': None, 'anonymous_id': 0, 'vote': 0, 'is_seen': True, 'is_starred': False, 'is_watched': True, 'glanced_at': '2024-05-30T05:11:28.997933+10:00', 'new_reply_count': 0, 'duplicate_title': None, 'user': {'id': 991487, 'role': 'user', 'name': 'David Akinboro', 'avatar': None, 'course_role': 'admin', 'tutorials': {'58877': ''}}}]

# #get thread
# print(ed.get_thread(4998433)['title'])
# print(ed.get_thread(955353)['answers'][1].keys()) #['document']

# #test the recurisve loop
# title  = "Word-word matrix".lower().strip()
# resp = find_similar_title(title, 0)

# print(resp)
# 619252

# [0.7405795913832682, 0.6628915657630697, 0.5554593077637721, 0.7887562879317197, 0.4167228026742793, 0.7397829442195785, 0.13198940700567446, 0.5805170067128785, 0.6600358680320267, 0.7662021871077065, 0.6624283195504911, 0.6632779134508807, 0.5349283998434504, 0.5086500124786336, 0.5586428436592376, 0.6749525434487247, 0.7998446732215979, 0.7765092733040937, 0.5505852802737947, 0.7971502763612859]
# Overall evaluation is 0.6284953252093083