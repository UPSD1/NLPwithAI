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
    ai_resp = thread_data['answers'][0]['document'] if thread_data['answers'] else None
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


# Skipped thread Nos [5047529, 5047488, 5047485, 5047478, 5047475, 5047463, 5047454, 5047450, 5047438, 5047436, 5047432, 5047430, 5047420, 5047416, 5045745, 5045731, 5045730, 5045710, 5045703, 5045702, 5045695, 5045648, 5045640, 5045638, 5045635, 5045528, 5045524, 5045522, 5045521, 5045519, 5045518, 5045509, 5045508, 5045499, 5045495, 5045479, 5045469, 5045462, 5045457, 5045442, 5045441, 5045438, 5045414, 5045409, 5045405, 5045404, 5045401, 5045396, 5044233, 5044217, 5044187, 5044184, 5044167, 5044165, 5044153, 5044152, 5044148, 5044142, 5044141, 5044137, 5044130, 5044129, 5044072, 5044071, 5044069, 5044063, 5044062, 5044053, 5044052, 5034326, 5034315, 5034304, 5034292, 5034276, 5034262, 5032107, 5029732, 5021557, 5021423, 5021417, 4998318, 4998285, 4996770, 4980951, 4977658, 4952434, 4947750]

# Overall evaluation is 0.43716627892783694
# Evaluation dict ---> {'Lectures': [0.2749011797620542, 0.7807057231525162, 0.6168924625037607, 0.7198190074497361, 0.8244542146974951, 0.5828072530189489, 0.5663813192944605, 0.6144063134063601, 0.29179335198793066, 0.3468457684588222, 0.14910638966214268, 0.6556399311773348, 0.11126034113701827, 0.6065876637697232, 0.15228365535029642, 0.4949924212402895, 0.3121773466194224, 0.2663394446166647, 0.452346156536367, 0.3053125367125733, 0.7300899526109166, 0.688813625371799, 0.5813559902052148, 0.7435410050141074, 0.8270759761531138, 0.7962793952775744], 'General': [0.5402143602469579, 0.5968875084416754, 0.10487446769621835, 0.2426266642398475, 0.4879375062789941, 0.45218154971533453, 0.5879629875164982, 0.6551357869732778, 0.5878128144587144, 0.12740192636438086, 0.20951633891271434, 0.09834733210735491, 0.29563261874525715, 0.2826645647521757, 0.4627299428177295, 0.5007872810635192, 0.544285474175624, 0.5190619077939659, 0.4008454099373283, 0.5445638375622726, 0.3671245949682651, 0.19679148951548234, 0.55827737422476, 0.24951923367721313, 0.31380761702818977, 0.24777106722659417, 0.6021133879246885, 0.528078716537093, 0.6145067383428955, 0.45206252021124654, 0.5802420440336585, 0.22608970195378159, 0.45014908339424153, 0.24566223177554453, 0.34285708886489025, 0.4581155242716064, 0.7335377340864151, 0.4680210426309303, 0.43580389422424337, 0.42684780675452405, 0.45174772672750363, 0.5654495769966927], 'Exams': [0.5371574612455445, 0.4966846899580539, 0.2982451642123278, 0.48783851728083577, 0.6529384864847767, 0.48462914786858957, 0.5518773982058338, 0.37711438317367435, 0.17637180909718636, 0.4883797555584182, 0.2048311113512068, 0.25827432078608165, 0.670093321729178, 0.2283995997702836, 0.37928405474292276, 0.6624382550097053, 0.6853207127227604, 0.42134371931264303, 0.5685652991474742, 0.31409265515089896, 0.6064315891779514, 0.5615686925928611, 0.5254100606173506, 0.44900953560934503, 0.46159044405572497, 0.27602566840629633, 0.21907552870022742, 0.6948503437870214, 0.5890793150759989, 0.06928473973719382, 0.42707701265320214, 0.214993514639903, 0.5958903509574877, 0.5970148446799066, 0.2270592493347517, 0.5004564125751642, 0.3068246640895698, 0.48628102308461396, 0.4044317067303364, 0.28408734457268237, 0.303731234623144, 0.5189923552171409, 0.2811684240727483, 0.4204549410316737, 0.29525338824738984, 0.3708331285071609, 0.345051645393458, 0.4073829009221888, 0.5977486327924023, 0.4673023241511175, 0.5998861762677403, 0.4134769613030916, 0.6869625413871968, 0.6030758524434737, 0.47149670142248895, 0.6598915814630797, 0.6289022855075317, 0.6285039847656914, 0.3720135443888671, 0.37502943033635433, 0.6724462139323557, 0.5437885980150616, 0.30065237271975104, 0.12719389000514664, 0.2907185411395283, 0.5319666858246213, 0.5065815382515552, 0.4997826221027762, 0.21074373603186805, 0.5830279409486212, 0.5219954890173478, 0.33957392854784796, 0.5123304737317815, 0.12248987444275516, 0.6536888436657462, 0.36599688963070054, 0.43577209720303506, 0.5442625321058552, 0.16098733557473757, 0.5484250725347732, 0.6090644919949503, 0.17832553587638864, 0.17946477826217408, 0.6112270274393599, 0.3667854150721131, 0.173555136659607, 0.1941417894361113, 0.5335956766132289, 0.360087644731024, 0.33086033897733064, 0.07738306228937804, 0.08569276150602356, 0.10029714695624226, 0.4523332148303218, 0.6558779294182961, 0.535460453323679, 0.20984517850324033, 0.5445308038306245], 'Critiques': [0.3055711647654042, 0.33167579547837395]}
# Final result: 
# {'Lectures': 0.5189310932764093, 'General': 0.4227630113135786, 'Exams': 0.42406564284975407, 'Critiques': 0.31862348012188907}