#**************************************library importation************************************************
from edapi import EdAPI #for connecting wih ed platform
from dotenv import load_dotenv #for getting enviroment variables
import json #used for json conversion
from pathlib import Path #used for path traversal
#*********************************************************************************************************

#***************************************Variable declaration*********************************************
course_id = 12801
priority_ids = {146168, 336652, 133439, 304091, 155425, 151439, 147345, 132907, 132929} #TA ids'
#********************************************************************************************************

#*****************************Initialization & Authorization********************************************** 
load_dotenv()
ed = EdAPI() # initialize Ed API
ed.login()  # authenticate user through the ED_API_TOKEN environment variable
user_info = ed.get_user_info() # retrieve user information; authentication is persisted to next API calls
user = user_info['user']
print(f"Hello {user['name']}!")
#**********************************************************************************************************

dataset = []
sys = "You are an university teaching assistant chatbot that guides students"
def create_dataset(offset):
    thread_lst = ed.list_threads(course_id, limit = 100, offset = offset, sort = "new" ) #gets 100 threads at a time
    
    #if we have no thread again
    if not thread_lst:
      return
    
    for i , thread in enumerate(thread_lst): #loop through each thread pulled in the thread list
        print(f"Thread {offset+i+1}:", end= " ")
        if (thread["type"].lower() == "announcement"): #if thread is an announcement skip
            print("Skipped because post is an announcement")
            if i == (len(thread_lst) - 1): #if that's the last thread in the list
                create_dataset(offset + len(thread_lst)) #pull the next batch of threads
            continue #move

        thread_id = thread['id'] #used to retrieve TA response
        content = thread['document'] #This is the question asked by student
        data = ed.get_thread(thread_id) #get further details of the thread

        if content: #if the content isn't an empty string  
            if len(data['answers']) >= 1: #check if we have multiple responses
                status = False
                for answer in data['answers']: #loop through the responses
                    if answer['user_id'] in priority_ids: #if the id of the user who responded is a TA
                        print()
                        status = True
                        result = answer['document'] #get the answer attached to the thread by TA
                        dataset.append(
                            {
                                "messages": 
                                [
                                    {
                                        "role": "system",
                                        "content": sys
                                    },
                                    {
                                        "role": "user",
                                        "content": content
                                    },
                                    {
                                        "role": "assistant",
                                        "content": result
                                    }
                                ]
                            }
                        )
                        break
                if not status:
                    print("Skipped because question was not answered by a staff")

            elif len(data['comments']) >= 1: #check if we have multiple comments
                status = False
                for answer in data['comments']: #loop through the responses
                    if answer['user_id'] in priority_ids: #if the id of the user who responded is a TA
                        print()
                        status = True
                        result = answer['document'] #get the answer attached to the thread by TA
                        dataset.append(
                            {
                                "messages": 
                                [
                                    {
                                        "role": "system",
                                        "content": sys
                                    },
                                    {
                                        "role": "user",
                                        "content": content
                                    },
                                    {
                                        "role": "assistant",
                                        "content": result
                                    }
                                ]
                            }
                        )
                        break
                if not status:
                    print("Skipped because question was not commented by a staff")

            else: #check if there was no answer or comment attached to the question
                print("Skipped because post was not answered nor commented")
        else:
            print("Skipped because there is no content probably just images")

        if i == (len(thread_lst) - 1): #if that's the last thread in the list
            create_dataset(offset + len(thread_lst)) #pull the next batch of threads
    return

def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [json.dumps(line) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))

def main():
    print("Creating Dataset ...")
    create_dataset(0)
    print("Writing Dataset to file ....")
    write_jsonl("data.jsonl", dataset)
    print("The End.")

if __name__ ==  "__main__":
    main()