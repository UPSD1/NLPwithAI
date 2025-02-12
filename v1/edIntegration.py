#----------------------------------Import Libraries-----------------------------------------
from edapi import EdAPI
from typing import TypedDict, NoReturn
import json
from edapi.types.api_types.endpoints.files import API_PostFile_Response
from edapi.types import EdAuthError, EdError
from requests.compat import urljoin
#-------------------------------------------------------------------------------------------

#----------------------------Variable Declaration and Initialization------------------------
ed = EdAPI()
API_BASE_URL = "https://us.edstem.org/api/"
#-------------------------------------------------------------------------------------------

#--------------------------------Function Declaration---------------------------------------
def _throw_error(message: str, error_content: bytes) -> NoReturn:
    """
    Throw an error with the given message and the error content.
    """
    error_json = json.loads(error_content)
    if error_json.get("code") == "bad_token":
        # auth error
        raise EdAuthError(
            {"message": f"Ed authentication error; {message}", "response": error_json}
        )

    # other error
    raise EdError({"message": message, "response": error_json})

class PostCommentParams(TypedDict, total=True):
    content: str
    is_anonymous: bool
    is_private: bool
    type: str

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

def test_post_thread(thread_id, params):
    thread_url = urljoin(API_BASE_URL, f"threads/{thread_id}/comments")
    response = ed.session.post(thread_url, json={"comment": params})
    return response

def post_comment(thread_id, params):
    post_comment_params = PostCommentParams(
        content=f"<document version=\"2.0\"><paragraph>{params}</paragraph></document>",
        # content=response,
        is_anonymous=False,
        is_private=True,
        type="answer"
    )

    response = test_post_thread(thread_id, post_comment_params)
    if response.ok:
        response_json: API_PostFile_Response = response.json()
        return response_json

    _throw_error(f"Failed to post comment {thread_id}.", response.content)
