from po_db import (
    save_application,
    save_chat_log,
    save_exploration_log,
    save_analytics_log
)
from pymongo import MongoClient

# Sample data for testing
sample_application = {
    'Current_Loan_Amount': 50000,
    'Credit_Score': 720,
    'Annual_Income': 85000,
    'Years_in_current_job': 5,
    'Term_Short_Term': 1,
    'Home_Ownership_Home_Mortgage': 0,
    'Home_Ownership_Own_Home': 1,
    'Home_Ownership_Rent': 0
}
sample_decision = "Approved"

sample_chat = {
    'prompt': "What are the requirements for a home loan?",
    'response': "To qualify for a home loan, you generally need a good credit score, steady income, and a manageable debt-to-income ratio."
}

sample_exploration = {
    'action': "Show Columns",
    'columns': ["Credit Score", "Annual Income"]
}

sample_analytics = {
    'filename': 'loan_data.csv',
    'plot_type': 'bar',
    'columns': ["Annual Income", "Loan Amount"]
}

# Save all test entries
print("Saving test data to MongoDB...")

save_application(sample_application, sample_decision)
print("✅ Application saved.")

save_chat_log(sample_chat['prompt'], sample_chat['response'])
print("✅ Chat log saved.")

save_exploration_log(sample_exploration['action'], sample_exploration['columns'])
print("✅ Exploration log saved.")

save_analytics_log(sample_analytics['filename'], sample_analytics['plot_type'], sample_analytics['columns'])
print("✅ Analytics log saved.")

# Optionally show all documents in each collection to verify
client = MongoClient("mongodb://localhost:27017/")
db = client["loan_applications"]

print("\n--- Application Entries ---")
for doc in db.applications.find():
    print(doc)

print("\n--- Chat Logs ---")
for doc in db.chat_logs.find():
    print(doc)

print("\n--- Exploration Logs ---")
for doc in db.exploration_logs.find():
    print(doc)

print("\n--- Analytics Logs ---")
for doc in db.analytics_logs.find():
    print(doc)
