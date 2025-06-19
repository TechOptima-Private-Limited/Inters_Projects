# from pymongo import MongoClient

# # Connect to local MongoDB instance
# client = MongoClient("mongodb://localhost:27017")

# # Use (or create) the database and collection
# db = client["credit_analyzer"]
# collection = db["qa_pairs"]

# # OPTIONAL: Clear old data (only if you're okay resetting it)
# collection.delete_many({})

# # Ensure full-text search index on 'question' field
# collection.create_index([("question", "text")])

# # 50 sample QA pairs
# qa_data = [
#     {
#         "question": "What is a credit score?",
#         "answer": "A credit score is a number ranging from 300 to 850 that indicates a person's creditworthiness."
#     },
#     {
#         "question": "How is a credit score calculated?",
#         "answer": "It's based on factors like payment history, credit utilization, length of credit history, types of credit, and new inquiries."
#     },
#     {
#         "question": "What is considered a good credit score?",
#         "answer": "A credit score above 700 is generally considered good."
#     },
#     {
#         "question": "How can I improve my credit score?",
#         "answer": "Pay bills on time, reduce debt, avoid new credit applications, and monitor your credit report."
#     },
#     {
#         "question": "Does checking my own credit lower my score?",
#         "answer": "No, checking your own credit report is a soft inquiry and doesn't affect your score."
#     },
#     {
#         "question": "What is credit utilization?",
#         "answer": "It refers to the amount of credit used compared to your total credit limit."
#     },
#     {
#         "question": "Should I close old credit cards?",
#         "answer": "Not necessarily. Keeping them open can help your credit history length and utilization."
#     },
#     {
#         "question": "How often is my credit score updated?",
#         "answer": "Credit scores can update monthly, depending on when lenders report data to bureaus."
#     },
#     {
#         "question": "What is a hard inquiry?",
#         "answer": "A hard inquiry occurs when a lender checks your credit report for a loan or credit application."
#     },
#     {
#         "question": "What is a soft inquiry?",
#         "answer": "A soft inquiry doesn't affect your credit score and happens during background or personal checks."
#     },
#     # Additional 40 dummy QAs
# ]

# qa_data += [
#     {"question": f"Sample credit-related question #{i}", "answer": f"Sample answer for credit-related question #{i}."}
#     for i in range(11, 51)
# ]

# # Insert into MongoDB
# collection.insert_many(qa_data)

# print("✅ Inserted 50 Q&A documents into 'credit_analyzer.qa_pairs'")


# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017")
# db = client["credit_analyzer"]
# collection = db["qa_pairs"]

# # Clear existing documents
# collection.delete_many({})

# # List of knowledge-base style content
# documents = [
#     {"content": "A credit score is a numerical expression representing the creditworthiness of a person, typically ranging from 300 to 850."},
#     {"content": "Credit scores are based on credit report information including payment history, amounts owed, length of credit history, credit mix, and new credit."},
#     {"content": "Payment history is the most important factor affecting credit scores, accounting for approximately 35% of the total score."},
#     {"content": "Keeping your credit utilization below 30% is recommended to maintain a healthy credit score."},
#     {"content": "Hard inquiries from applying for credit can lower your score slightly and stay on your report for up to 2 years."},
#     {"content": "Soft inquiries, such as checking your own score or pre-approvals, do not affect your credit score."},
#     {"content": "A good credit score is generally considered to be 700 or above by most lenders."},
#     {"content": "Late payments, collections, and charge-offs are examples of negative marks that hurt your credit."},
#     {"content": "You can dispute errors in your credit report with the credit bureaus if you find inaccurate information."},
#     {"content": "Paying your bills on time is the single most effective way to build or maintain your credit score."},
#     {"content": "Opening several new credit accounts in a short period can be risky and lower your score."},
#     {"content": "Credit mix refers to having different types of credit accounts like credit cards, auto loans, and mortgages."},
#     {"content": "Older credit accounts contribute positively to your credit score by extending your credit history length."},
#     {"content": "Closing a credit card can affect your utilization and average age of accounts, potentially lowering your score."},
#     {"content": "Your debt-to-income ratio helps lenders understand your ability to repay debt."},
#     {"content": "Credit reports are typically updated every 30–45 days as lenders report your account activity."},
#     {"content": "Negative items like bankruptcies and defaults can remain on your credit report for 7–10 years."},
#     {"content": "Secured credit cards require a deposit and are useful for building or rebuilding credit."},
#     {"content": "Co-signing a loan makes you equally responsible for the debt, affecting your credit score."},
#     {"content": "Installment credit includes auto loans and mortgages, while revolving credit includes credit cards."},
#     {"content": "You can improve your credit by keeping balances low, making timely payments, and avoiding new debt."},
#     {"content": "Utility bills and rent payments may be included in your credit history through third-party services."},
#     {"content": "A credit freeze restricts access to your report, protecting you from identity theft."},
#     {"content": "Too many hard inquiries in a short time frame suggest financial distress and may reduce your score."},
#     {"content": "Lenders use FICO and VantageScore models to assess your creditworthiness differently."},
#     {"content": "Income is not directly included in credit scores, but it affects a lender's ability to offer credit."},
#     {"content": "You can check your credit report for free once per year at AnnualCreditReport.com."},
#     {"content": "The snowball method prioritizes paying off the smallest debt first, while the avalanche targets highest interest rate."},
#     {"content": "Credit bureaus include Experian, Equifax, and TransUnion, each maintaining separate reports."},
#     {"content": "Disputing inaccuracies promptly can help remove errors that drag down your score."},
#     {"content": "Lenders look at both your credit report and score to evaluate loan applications."},
#     {"content": "Student loans impact your credit similarly to other installment loans — missed payments hurt, on-time helps."},
#     {"content": "Collections accounts reflect unpaid debts sent to agencies and significantly damage credit scores."},
#     {"content": "Using auto-pay can help ensure on-time payments and build a positive credit history."},
#     {"content": "Closing the oldest account can shorten your credit history and may hurt your score."},
#     {"content": "A thin credit file means you don't have enough credit history, which makes scoring more difficult."},
#     {"content": "Authorized users inherit part of the credit behavior from the primary account holder."},
#     {"content": "Bankruptcy can lower your score by 200+ points and remains for up to 10 years."},
#     {"content": "Defaulting on a loan appears on your report and significantly reduces your score."},
#     {"content": "A charge-off is when a creditor writes off your debt as uncollectible and reports it negatively."},
#     {"content": "Some employers check credit reports (not scores) during hiring for roles involving financial trust."},
#     {"content": "Missed mortgage payments can trigger foreclosure and major drops in credit score."},
#     {"content": "Prepaid cards generally don’t report to credit bureaus and don’t build credit history."},
#     {"content": "Rent reporting services can help add rental payments to your credit file."},
#     {"content": "Late payments stay on your report for 7 years, even if you later pay the debt."},
#     {"content": "You can request a credit limit increase to lower your utilization ratio if you manage credit well."},
#     {"content": "Credit card utilization is calculated per card and overall — both matter for your score."},
#     {"content": "Secured loans are backed by collateral; unsecured loans depend solely on creditworthiness."},
#     {"content": "Foreclosure is the legal process when a lender takes your property for failing to repay your mortgage."},
#     {"content": "Your credit report does not include your credit score; it's calculated separately by scoring models."},
#     {"content": "Regular monitoring can help detect fraud and errors before they seriously affect your credit."}
# ]

# # Insert into MongoDB
# collection.insert_many(documents)

# # Create a full-text search index on 'content'
# collection.create_index([("content", "text")])

# print("✅ Inserted 50 credit knowledge documents and created text index.")




from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["credit_analyzer"]
collection = db["credit_docs"]

# Optional: Clear existing documents
collection.delete_many({})
print("🧹 Cleared existing documents.")

# Sample content documents
documents = [
    {"content": "A credit score is a numerical expression representing the creditworthiness of a person."},
    {"content": "Payment history and credit utilization are key components of a credit score."},
    {"content": "Hard inquiries slightly reduce your score and stay on the report for up to 2 years."},
    {"content": "A good credit score is considered to be 700 or higher."},
    {"content": "Keeping credit utilization below 30% helps maintain a healthy score."},
    {"content": "Your debt-to-income ratio helps lenders understand your ability to repay debt."},
    {"content": "Credit reports are typically updated every 30–45 days as lenders report your account activity."},
    {"content": "Negative items like bankruptcies and defaults can remain on your credit report for 7–10 years."},
    {"content": "Secured credit cards require a deposit and are useful for building or rebuilding credit."},
    {"content": "Co-signing a loan makes you equally responsible for the debt, affecting your credit score."},
    {"content": "Installment credit includes auto loans and mortgages, while revolving credit includes credit cards."},
    {"content": "You can improve your credit by keeping balances low, making timely payments, and avoiding new debt."},
    {"content": "Utility bills and rent payments may be included in your credit history through third-party services."},
]

# Insert new documents
insert_result = collection.insert_many(documents)
print(f"✅ Inserted {len(insert_result.inserted_ids)} documents.")

# Ensure text index exists
index_exists = any(
    index.get('key', [])[0][0] == 'content' and index.get('key', [])[0][1] == 'text'
    for index in collection.index_information().values()
)

if not index_exists:
    collection.create_index([("content", "text")], name="content_text")
    print("🔠 Text index on 'content' created.")
else:
    print("ℹ️ Text index on 'content' already exists.")
