
# Keycloak Authenticated API Client in Python

This Python project demonstrates how to authenticate with a **Keycloak** server using the **Resource Owner Password Credentials (ROPC)** flow and then call a protected API endpoint using the obtained token.

---

## 🔧 Requirements

- Python 3.7+
- A working Keycloak server
- A client in Keycloak configured for direct access grants (ROPC)

---

## 📁 Project Structure

```

.
├── main.py             # Main script to authenticate and call the API
├── .env                # Stores environment variables (credentials, URLs)
├── requirements.txt    # Python dependencies
└── README.md           # This file

```

---

## 📦 Setup

1. **Clone this repository** (or copy files into your project directory):

2. **Create and configure your `.env` file:**

```

KEYCLOAK\_URL=[https://keycloak.azure.openiotlab.eu/](https://keycloak.azure.openiotlab.eu/)
KEYCLOAK\_REALM=irritre
KEYCLOAK\_CLIENT\_ID=backend-ws
KEYCLOAK\_USERNAME=[rsilvestri@fbk.eu](mailto:rsilvestri@fbk.eu)
KEYCLOAK\_PASSWORD=zwHkLQa2hvBSzCK
API\_URL=[https://irritre-sensor.azure.openiotlab.eu/api/device-locations](https://irritre-sensor.azure.openiotlab.eu/api/device-locations)

````

> ⚠️ **Never commit your `.env` file to version control** – it contains sensitive credentials.

3. **Install dependencies:**

```bash
pip install -r requirements.txt
````

---

## ▶️ Run the Script

```bash
python main.py
```

If successful, it will:

* Authenticate with Keycloak using your credentials.
* Retrieve an access token.
* Use that token to make a GET request to the protected API endpoint.
* Print the response.

---

## 🔐 Notes

* This project uses the **password grant** type, which is suitable for trusted applications.
* Make sure the Keycloak client has `Direct Access Grants Enabled`.
* If the API token expires, you may need to implement a **refresh token** flow.

---

## 🧼 Security Tip

Use environment variables or secret management tools for production environments. Avoid hardcoding credentials.

---

## 📜 License

This project is provided as-is for educational and integration purposes.
