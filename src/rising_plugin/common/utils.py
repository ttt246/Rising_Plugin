import os
import json

# env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
FIREBASE_ENV = os.getenv("FIREBASE_SERVICE_ACCOUNT_TEST3_83FFC")

# firebase
FIREBASE_STORAGE_ROOT = "images/"
FIREBASE_STORAGE_BUCKET = "test3-83ffc.appspot.com"

# pinecone
PINECONE_NAMESPACE = "risinglangchain-namespace"
PINECONE_INDEX_NAME = "risinglangchain-index"


# indexes of relatedness of embedding
COMMAND_SMS_INDEXS = [4, 5]
COMMAND_BROWSER_OPEN = [10]


class ProgramType:
    BROWSER = "browser"
    ALERT = "alert"
    IMAGE = "image"
    SMS = "sms"
    CONTACT = "contact"
    MESSAGE = "message"


# validate json format
def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True
