# po_db.py

from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["loan_applications"]

# Collections
applications_collection = db["applications"]
chat_logs_collection = db["chat_logs"]
exploration_logs_collection = db["exploration_logs"]
analytics_logs_collection = db["analytics_logs"]

# 1. Save a single loan application prediction
def save_application(data, decision):
    record = {
        "input": data,
        "decision": decision,
        "timestamp": datetime.now()
    }
    applications_collection.insert_one(record)

# 2. Save OptGPT chat
def save_chat_log(question, response):
    record = {
        "question": question,
        "response": response,
        "timestamp": datetime.now()
    }
    chat_logs_collection.insert_one(record)

# 3. Save Data Exploration actions
def save_exploration_log(user_action, columns=None):
    record = {
        "action": user_action,
        "columns": columns if columns else [],
        "timestamp": datetime.now()
    }
    exploration_logs_collection.insert_one(record)

# 4. Save Analytics (uploaded dataset + plot info)
def save_analytics_log(file_name, plot_type, selected_columns):
    record = {
        "file_name": file_name,
        "plot_type": plot_type,
        "columns": selected_columns,
        "timestamp": datetime.now()
    }
    analytics_logs_collection.insert_one(record)

print("[po_db.py] MongoDB connection successful and backend functions ready.")
