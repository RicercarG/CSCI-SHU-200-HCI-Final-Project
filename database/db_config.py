import os
from dotenv import load_dotenv
import streamlit as st
import mysql.connector
from mysql.connector import Error

# Load environment variables from .env file
load_dotenv()

# Database connection details from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "electric_vehicle_assistant")

def get_database_connection():
    """
    Create a secure database connection using environment variables
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            use_pure=True,
            autocommit=True,
            ssl_disabled=False,  # Enable SSL for secure connections
            connection_timeout=10
        )
        return connection
    except Error as e:
        st.error(f"Database connection failed: {e}")
        return None

def init_streamlit_connection():
    """
    Initialize the Streamlit SQL connection with secure parameters
    """
    if "db_connection" not in st.session_state:
        # Create a database URL for SQLAlchemy
        db_url = f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        try:
            # Simple connection approach without complex parameters
            st.session_state.db_connection = st.connection("mysql", type="sql", url=db_url)
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            # Fallback to a simpler connection without SSL
            try:
                fallback_url = f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                st.session_state.db_connection = st.connection("mysql", type="sql", url=fallback_url)
            except Exception as e2:
                st.error(f"Fallback connection also failed: {e2}")
                return None
    
    return st.session_state.db_connection 