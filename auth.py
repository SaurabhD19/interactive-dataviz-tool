import streamlit as st
import json
import os
from config import CREDENTIALS_FILE


def load_credentials() -> dict:
    """
    Load user credentials from a JSON file.

    Returns:
        dict: Dictionary of usernames and passwords.
    """
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    else:
        return {}


def save_credentials(users: dict) -> None:
    """
    Save user credentials to a JSON file.

    Args:
        users (dict): Dictionary of usernames and passwords.
    """
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(users, f)


# Load existing credentials at the start
users = load_credentials()


def login_page() -> None:
    """
    Display the login page.
    """
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")


def register_page() -> None:
    """
    Display the registration page.
    """
    st.title("Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if username in users:
            st.error("Username already exists!")
        else:
            users[username] = password
            save_credentials(users)
            st.success("Registration successful! Please log in.")
            st.rerun()
