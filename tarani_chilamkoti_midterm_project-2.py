

import os
import itertools
import pandas as pd
from collections import defaultdict
from mlxtend.frequent_patterns import apriori, association_rules
import time

# Ask user to provide the local path to the dataset
file_path = input("Enter the full path to your dataset CSV file: ").strip()

# Verify the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

# Get minimum support & confidence from user
try:
    min_support = float(input("Enter minimum support value (0-1): "))
    min_confidence = float(input("Enter minimum confidence value (0-1): "))

    if not (0 <= min_support <= 1 and 0 <= min_confidence <= 1):
        raise ValueError("Values must be between 0 and 1.")
except ValueError:
    print("Invalid input. Please enter numbers between 0 and 1.")
    exit()

# Load dataset and convert transactions
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    transactions = df["Transaction"].apply(lambda x: set(map(str.strip, x.split(",")))).tolist()
    return transactions

# Generate itemsets
def generate_itemsets(items, k):
    return list(itertools.combinations(items, k))

# Calculate support
def calculate_support(transactions, itemsets):
    support_count = defaultdict(int)
    total_transactions = len(transactions)

    for transaction in transactions:
        for itemset in itemsets:
            if set(itemset).issubset(transaction):
                support_count[itemset] += 1

    return {itemset: count / total_transactions for itemset, count in support_count.items()}

# Generate association rules using brute force
def generate_brute_force_rules(transactions, min_support, min_confidence):
    items = set(item for transaction in transactions for item in transaction)
    support_dict = defaultdict(int)
    total_transactions = len(transactions)

    for transaction in transactions:
        for item in transaction:
            support_dict[(item,)] += 1

    for key in support_dict:
        support_dict[key] /= total_transactions

    rules = []
    for k in range(1, min(4, len(items)) + 1):
        itemsets = generate_itemsets(items, k)
        for itemset in itemsets:
            support_count = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
            support = support_count / total_transactions
            if support >= min_support:
                for i in range(1, len(itemset)):
                    for antecedent in itertools.combinations(itemset, i):
                        consequent = tuple(set(itemset) - set(antecedent))
                        if consequent:
                            antecedent_support = support_dict.get(antecedent, 0)
                            if antecedent_support > 0:
                                confidence = support / antecedent_support
                                if confidence >= min_confidence:
                                    rules.append({
                                        "Antecedents": set(antecedent),
                                        "Consequents": set(consequent),
                                        "Support": support,
                                        "Confidence": confidence
                                    })
    return rules

# Apriori Algorithm
def apriori_algorithm(transactions, min_support, min_confidence):
    all_items = sorted(set(item for transaction in transactions for item in transaction))
    encoded_data = [{item: (item in transaction) for item in all_items} for transaction in transactions]
    df_encoded = pd.DataFrame(encoded_data)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules[['antecedents', 'consequents', 'support', 'confidence']]

# Process dataset
print(f"\nProcessing dataset: {file_path}")
transactions = load_dataset(file_path)

# Run Brute Force
start_time = time.time()
brute_force_rules = generate_brute_force_rules(transactions, min_support, min_confidence)
print(f"Brute Force Time: {time.time() - start_time:.2f} seconds")

# Run Apriori
start_time = time.time()
apriori_rules = apriori_algorithm(transactions, min_support, min_confidence)
print(f"Apriori Time: {time.time() - start_time:.2f} seconds")

# Display results
if brute_force_rules:
    print("\nBrute Force Association Rules:")
    print(pd.DataFrame(brute_force_rules))
else:
    print("\nNo strong association rules found using Brute Force.")

if not apriori_rules.empty:
    print("\nApriori Association Rules:")
    print(apriori_rules)
else:
    print("\nNo strong association rules found using Apriori.")