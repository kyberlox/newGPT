import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('key')
organization = os.getenv('organization')
project_id = os.getenv('project_id')

class Key:
    def __init__(self):
        self.key = key
        self.organization = organization
        self.project_id = project_id
