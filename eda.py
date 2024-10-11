import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your datasets (replace file paths with actual locations if needed)
transaction_data = pd.read_csv('path_to_transaction_data.csv')
customer_data = pd.read_csv('path_to_customer_data.csv')
product_info = pd.read_csv('path_to_product_info.csv')

# Merging the data
merged_data = pd.merge(transaction_data, customer_data, on='Customer ID', how='left')
merged_data = pd.merge(merged_data, product_info, on='StockCode', how='left')

# Add a 'Revenue' column
merged_data['Revenue'] = merged_data['Quantity'] * merged_data['Price']

# 1a. Customer-level summary
customer_summary = merged_data.groupby('Customer ID').agg(
    num_transactions=('Invoice', 'nunique'),
    total_quantity=('Quantity', 'sum'),
    total_revenue=('Revenue', 'sum')
).reset_index()

# 1a. Item-level summary
item_summary = merged_data.groupby('StockCode').agg(
    total_sales=('Quantity', 'sum'),
    total_revenue=('Revenue', 'sum')
).reset_index()

# 1a. Transaction-level summary
transaction_summary = merged_data.groupby('Invoice').agg(
    total_quantity=('Quantity', 'sum'),
    total_revenue=('Revenue', 'sum')
).reset_index()

# 1d. Visualizations
sns.set_theme(style="whitegrid")

# Plot total quantity sold per customer
plt.figure(figsize=(10, 6))
sns.barplot(x='Customer ID', y='total_quantity', data=customer_summary.sort_values('total_quantity', ascending=False))
plt.title('Total Quantity Sold per Customer')
plt.xticks(rotation=45)
plt.show()

# Plot total revenue per product (item-level analysis)
plt.figure(figsize=(10, 6))
sns.barplot(x='StockCode', y='total_revenue', data=item_summary.sort_values('total_revenue', ascending=False))
plt.title('Total Revenue per Product')
plt.xticks(rotation=45)
plt.show()

# Visualize transaction-level summary (Total revenue per transaction)
plt.figure(figsize=(10, 6))
sns.histplot(transaction_summary['total_revenue'], bins=10, kde=True)
plt.title('Distribution of Total Revenue per Transaction')
plt.show()
