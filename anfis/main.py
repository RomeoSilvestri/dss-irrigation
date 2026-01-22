import os
import requests
from dotenv import load_dotenv

load_dotenv()

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL").rstrip("/")
REALM = os.getenv("KEYCLOAK_REALM")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
USERNAME = os.getenv("KEYCLOAK_USERNAME")
PASSWORD = os.getenv("KEYCLOAK_PASSWORD")
API_URL = os.getenv("API_URL")

def get_access_token():
    token_url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "client_id": CLIENT_ID,
        "username": USERNAME,
        "password": PASSWORD
    }
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def call_protected_api(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(API_URL, headers=headers)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    token = get_access_token()
    data = call_protected_api(token)
    print("API Response:")
    print(data)
