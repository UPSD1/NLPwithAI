#----------------------------------Import Libraries-----------------------------------------
from edapi import EdAPI
#-------------------------------------------------------------------------------------------

#----------------------------Variable Declaration and Initialization------------------------
ed = EdAPI()
#-------------------------------------------------------------------------------------------

#--------------------------------Function Declaration---------------------------------------
def authenticate(verbose: bool = False):
    ed = EdAPI()
    ed.login()
    user_info = ed.get_user_info()

    if verbose:
        user_info = ed.get_user_info()
        user = user_info['user']
        print(f"Hello {user['name']}!")

    if user_info:
        return True, user_info
    
    return False

