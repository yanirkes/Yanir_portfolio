from BinaryTree import Tree
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
from typing import Any, Dict
import pickle
import os

# intiate a REST api with fast api
app = FastAPI()

# wil help us decoding into JSON the incoming request
JSONObject = Dict[Any, Any]

def create_new_tree():
    '''intiale a tree'''
    return Tree()

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

def check_request_values(request):
    '''
    This function created for checking the values which were sent by a user - making sure they are set properly
        if one of the follows raise, a 400 message will be sent (as request in the requirement doc)
    '''
    if ((not isinstance(request['tree'], dict)) and (request['tree'] is not None) ) :
        raise HTTPException(status_code=400 ,detail="tree type is Null or a dict")

    if set(request.keys()) != {'value', 'tree'}:
        raise HTTPException(status_code=400 ,detail="keys insert are not correct - should be value or tree")

    if request['tree'] is not None:
        if set(request['tree'].keys()) != {'value','left','right'}:
            raise HTTPException(status_code=400 ,detail="tree dict keys are incorrect - should be 'value','left','right'")

    if not isinstance(request['value'], int):
        raise HTTPException(status_code=400 ,detail="value must integer")


def save_object(obj, filename):
    '''save an object in a pickle'''
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, -1)
    return

def open_object(filename):
    '''open a pickle object'''
    with open(filename, 'rb') as inp:
        treeObj = pickle.load(inp)
    return treeObj

@app.post("/insert")
async def create_tree(request: JSONObject = None):
    '''main func that in charge of getting a post request by a user.
        checking if there is a tree - if not - create one and save it as a pickle.
        If there is already a tree - retrieve it from the pickled file, than add a node according the posted values. finally, update
        the pickle file with updated tree and send it to the user.
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))+'\myTreeObj.pkl'
    check_request_values(request)
    if request["tree"] == None:
        myTree = create_new_tree()
        myTree.add(request["value"])
        save_object(myTree, dir_path)
        res_tree = myTree.return_tree()
        return res_tree
    else:
        myTree = open_object(dir_path)
        myTree.add(request["value"])
        save_object(myTree, dir_path)
        res_tree = myTree.return_tree()
        return res_tree

if __name__ == "__main__":
    uvicorn.run(app, port=5000, host="0.0.0.0")


