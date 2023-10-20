import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
app = Flask('app')

ordered_df = pd.read_csv('ordered_df.csv')



# def fetch_brand(brand):

#     # Create an empty dictionary to store brands and their associated sets of unique products
#     brand_product_dict = {}

#     # Create a dictionary to keep track of offers for each brand
#     brand_offers = {}

#     # Iterate through the rows in the dataframe
#     for _, row in ordered_df.iterrows():
#         brand = row['BRAND']
#         product = row['Product']
#         offer = row['OFFER']

#         # Check if an offer exists for the brand (ignore rows where offer is NaN)
#         if not pd.isna(offer):
#             # Check if the brand has an entry in the dictionary
#             if brand not in brand_product_dict:
#                 brand_product_dict[brand] = set()  # Use a set to store unique products

#             # Add the product to the brand's set of unique products
#             brand_product_dict[brand].add(product)

#         # Keep track of offers for each brand
#         if brand not in brand_offers:
#             brand_offers[brand] = []

#         # Add the offer to the brand's list of offers
#         brand_offers[brand].append(offer)


#     def get_offers_for_brand(brand_name):
#         # Filter the dataframe for the specified brand
#         filtered_df = ordered_df[ordered_df['BRAND'] == brand_name]

#         if filtered_df['OFFER'].isna().all():
#             # Extract the products associated with the brand without offers
#             products_for_brand_without_offers = set(filtered_df['Product'].unique())

#             offer_similarities = []

#             for other_brand, other_products in brand_product_dict.items():
#                 intersection = len(products_for_brand_without_offers.intersection(other_products))
#                 union = len(products_for_brand_without_offers.union(other_products))
#                 similarity = intersection / union

#                 if similarity > 0:
#                     offer_similarities.append((other_brand, similarity))

#             if not offer_similarities:
#                 return {"Similar Brand":None, 'Offers':"No similar brands with offers found.", "Similarity Score": None}

#             sorted_similar_brands = sorted(offer_similarities, key=lambda x: x[1], reverse=True)

#             similar_brand_info = []
#             for similar_brand, similarity_score in sorted_similar_brands:
#                 similar_brand_offers = brand_product_dict.get(similar_brand, [])
#                 similar_brand_actual_offers = ordered_df[ordered_df['BRAND'] == similar_brand]['OFFER'].unique()
#                 similar_brand_info.append({
#                     "Similar Brand": similar_brand,
#                     "Similarity Score": similarity_score,
#                     "Offers": similar_brand_actual_offers.tolist() if similar_brand_actual_offers.size > 0 else ["No offers"]
#                 })

#             return similar_brand_info
#         else:
#             offers_for_brand = filtered_df['OFFER'].unique()
#             unique_offers = list(set(offers_for_brand))

#             if unique_offers:
#                 return {"Similar Brand":offers_for_brand, 'Offers':[offer_line for offer_line in unique_offers], "Similarity Score": 1}

#             else:
#                 return {"Similar Brand":None, 'Offers':"No similar brands with offers found.", "Similarity Score": None}   
               
#     top_similar_brands = get_offers_for_brand(str(brand))

#     return top_similar_brands

def fetch_brand(x):
        
            # Create an empty dictionary to store brands and their associated sets of unique products
    brand_product_dict = {}

    # Create a dictionary to keep track of offers for each brand
    brand_offers = {}

    # Iterate through the rows in the dataframe
    for _, row in ordered_df.iterrows():
        brand = row['BRAND']
        product = row['Product']
        offer = row['OFFER']

        # Check if an offer exists for the brand (ignore rows where offer is NaN)
        if not pd.isna(offer):
            # Check if the brand has an entry in the dictionary
            if brand not in brand_product_dict:
                brand_product_dict[brand] = set()  # Use a set to store unique products

            # Add the product to the brand's set of unique products
            brand_product_dict[brand].add(product)

        # Keep track of offers for each brand
        if brand not in brand_offers:
            brand_offers[brand] = []

        # Add the offer to the brand's list of offers
        brand_offers[brand].append(offer)

    # Filter brands where all offers are NaN
    brands_with_nan_offers = [brand for brand, offers in brand_offers.items() if all(pd.isna(offer) for offer in offers)]

    # Now, brand_product_dict contains brands as keys and sets of unique products as values
    # And brands_with_nan_offers contains brands with all NaN offers

    def get_offers_for_brand(brand_name):
        # Create an empty dictionary to store brands and their associated sets of unique products
        brand_product_dict = {}

        # Create a dictionary to keep track of offers for each brand
        brand_offers = {}

        # Iterate through the rows in the dataframe
        for _, row in ordered_df.iterrows():
            brand = row['BRAND']
            product = row['Product']
            offer = row['OFFER']

            # Check if an offer exists for the brand (ignore rows where offer is NaN)
            if not pd.isna(offer):
                # Check if the brand has an entry in the dictionary
                if brand not in brand_product_dict:
                    brand_product_dict[brand] = set()  # Use a set to store unique products

                # Add the product to the brand's set of unique products
                brand_product_dict[brand].add(product)

            # Keep track of offers for each brand
            if brand not in brand_offers:
                brand_offers[brand] = []

            # Add the offer to the brand's list of offers
            brand_offers[brand].append(offer)

        # Filter brands where all offers are NaN
        brands_with_nan_offers = [brand for brand, offers in brand_offers.items() if all(pd.isna(offer) for offer in offers)]

        # Filter the dataframe for the specified brand
        filtered_df = ordered_df[ordered_df['BRAND'] == brand_name]

        if filtered_df['OFFER'].isna().all():
            # Extract the products associated with the brand without offers
            products_for_brand_without_offers = set(filtered_df['Product'].unique())

            offer_similarities = []

            for other_brand, other_products in brand_product_dict.items():
                intersection = len(products_for_brand_without_offers.intersection(other_products))
                union = len(products_for_brand_without_offers.union(other_products))
                similarity = intersection / union

                if similarity > 0:
                    offer_similarities.append((other_brand, similarity))

            if not offer_similarities:
                return {"Similar Brand":[None], 'Offers':["No similar brands with offers found."], "Similarity Score": None}
            sorted_similar_brands = sorted(offer_similarities, key=lambda x: x[1], reverse=True)

            similar_brand_info = []
            for similar_brand, similarity_score in sorted_similar_brands:
                similar_brand_offers = brand_offers.get(similar_brand, [])
                similar_brand_actual_offers = ordered_df[ordered_df['BRAND'] == similar_brand]['OFFER'].unique()
                similar_brand_info.append({
                    "Similar Brand": similar_brand,
                    "Similarity Score": similarity_score,
                    "Offers": similar_brand_actual_offers.tolist() if similar_brand_actual_offers.size > 0 else ["No offers"]
                })
            combined_dict = {}

            for item in similar_brand_info:
                for key, value in item.items():
                    if key not in combined_dict:
                        combined_dict[key] = []
                    combined_dict[key].append(value)

            return combined_dict
        else:
            offers_for_brand = filtered_df['OFFER'].unique()
            unique_offers = list(set(offers_for_brand))

            if unique_offers:
                return {"Similar Brand":offers_for_brand, 'Offers':[offer_line for offer_line in unique_offers], "Similarity Score": 1}

            else:
                return {"Similar Brand":[None], 'Offers':["No similar brands with offers found."], "Similarity Score": None}   
            
    return get_offers_for_brand(x)


def fetch_category(category):


    # Create an empty dictionary to store product categories and their associated sets of unique products
    category_product_dict = {}

    # Create a dictionary to keep track of offers for each category
    category_offers = {}

    # Iterate through the rows in the dataframe
    for _, row in ordered_df.iterrows():
        product_category = row['Product_category']
        product = row['Product']
        offer = row['OFFER']

        # Check if an offer exists for the product category (ignore rows where offer is NaN)
        if not pd.isna(offer):
            # Check if the product category has an entry in the dictionary
            if product_category not in category_product_dict:
                category_product_dict[product_category] = set()  # Use a set to store unique products

            # Add the product to the product category's set of unique products
            category_product_dict[product_category].add(product)

        # Keep track of offers for each category
        if product_category not in category_offers:
            category_offers[product_category] = []

        # Add the offer to the category's list of offers
        category_offers[product_category].append(offer)

    # # Filter categories where all offers are NaN
    # categories_with_nan_offers = [category for category, offers in category_offers.items() if all(pd.isna(offer) for offer in offers)]

        # Define the function to compute category similarity and get offers
    def get_similar_categories_and_offers_with_similarity(input_category, top_n=5):
        # Tokenize and encode the input category
        input_tokens = tokenizer(input_category, return_tensors='pt', padding=True, truncation=True)

        # Get the BERT embeddings for the input category
        with torch.no_grad():
            input_category_embedding = model(**input_tokens).last_hidden_state.mean(dim=1)  # Mean pooling

        # Calculate cosine similarity with all categories that have offers
        category_similarities = {}
        for category in ordered_df[~ordered_df['OFFER'].isna()]['Product_category'].unique():
            category_tokens = tokenizer(category, return_tensors='pt', padding=True, truncation=True)
            category_embedding = model(**category_tokens).last_hidden_state.mean(dim=1)
            similarity = cosine_similarity(input_category_embedding.detach().numpy(), category_embedding.detach().numpy())
            category_similarities[category] = similarity[0][0]

        # Sort categories by similarity score in descending order
        sorted_similar_categories = sorted(category_similarities.items(), key=lambda x: x[1], reverse=True)

        # Select the top N similar categories
        top_similar_categories = sorted_similar_categories[:top_n]

        # Retrieve and return the offers and their associated similarity scores for the selected similar categories
        # similar_category_offers = []
        similar_category_offers = []

        for similar_category, similarity_score in top_similar_categories:
            offers = ordered_df[(ordered_df['Product_category'] == similar_category) & (~ordered_df['OFFER'].isna())]['OFFER'].unique()
            similar_category_offers.append((similar_category, offers.tolist(), similarity_score))

        if not similar_category_offers:
            return ["No similar categories"]
        return similar_category_offers

    def get_similar_categories_and_offers(category_name):
        # Check if the category is in category_product_dict
        if category_name in category_product_dict:
            # If the category has non-NaN offers, return them
            category_offers = ordered_df[(ordered_df['Product_category'] == category_name) & (~ordered_df['OFFER'].isna())]['OFFER'].unique()
            return category_offers  # Just return the actual offers

        # Category not found in category_product_dict, find similar categories
        similarity_scores = {}

        # Get the set of products associated with the category (excluding NaN offers)
        products_for_category = set(ordered_df[(ordered_df['Product_category'] == category_name) & (~ordered_df['OFFER'].isna())]['Product'].unique())

        for other_category, other_products in category_product_dict.items():
            # Skip the same category
            if other_category == category_name:
                continue

            # Calculate Jaccard similarity between the two categories
            intersection = len(products_for_category.intersection(other_products))
            union = len(products_for_category.union(other_products))

            # Handle cases where both categories have no associated products (all NaN offers)
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union

            similarity_scores[other_category] = similarity

        # Sort similar categories by similarity score in descending order
        sorted_similar_categories = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out categories with a similarity score of 0
        similar_category_offers = [(similar_category, ordered_df[(ordered_df['Product_category'] == similar_category) & (~ordered_df['OFFER'].isna())]['OFFER'].unique()) for similar_category, similarity_score in sorted_similar_categories if similarity_score != 0]

        # If there are similar category offers, return them
        if similar_category_offers:
            return similar_category_offers        # If there are no similar category offers, return "No offers"
        
        return ["No offers"]
    
    def get_similar_categories_and_offers_final(category_name, top_n=5):
    # Check if the category exists in the ordered_df
        if category_name in ordered_df['Product_category'].unique():
            # Call the first function for existing categories
            similar_offers = get_similar_categories_and_offers(category_name)
            return [similar_offers, False]
            # for similar_category, offers in similar_offers:
            #     yield similar_category, offers, None
        else:
            # Call the second function for non-existing categories
            similar_offers = get_similar_categories_and_offers_with_similarity(category_name, top_n)
            return [similar_offers, True]            
            # for similar_category, offers, similarity_score in similar_offers:
            #     yield similar_category, offers, similarity_score



    # Example usage:
    category_name = category  # Replace with the desired category name
    similar_offers = get_similar_categories_and_offers_final(category_name)
    return similar_offers




def merchant_offers(merchant):

# Create a list to store retailers with all NaN offers
    retailers_with_nan_offers = []

    # Create a list to store retailers without NaN or not all NaN offers
    retailers_with_non_nan_offers = []

    # Iterate through the rows in the dataframe
    for _, row in ordered_df.iterrows():
        retailer = row['RETAILER']
        offer = row['OFFER']

        # Check if an offer exists for the retailer (ignore rows where offer is NaN)
        if not pd.isna(offer):
            # Add the retailer to the list of retailers without NaN or not all NaN offers
            if retailer not in retailers_with_non_nan_offers:
                retailers_with_non_nan_offers.append(retailer)
        else:
            # Add the retailer to the list of retailers with all NaN offers
            if retailer not in retailers_with_nan_offers:
                retailers_with_nan_offers.append(retailer)

    # Now, retailers_with_nan_offers contains retailers with all NaN offers
    # retailers_with_non_nan_offers contains retailers without NaN or not all NaN offers


    # Create an empty dictionary to store retailers and their associated sets of unique products
    retailer_product_dict = {}

    # Create a dictionary to keep track of offers for each retailer
    retailer_offers = {}

    # Iterate through the rows in the dataframe
    for _, row in ordered_df.iterrows():
        retailer = row['RETAILER']
        product = row['Product']
        offer = row['OFFER']

        # Check if an offer exists for the retailer (ignore rows where offer is NaN)
        if not pd.isna(offer):
            # Check if the retailer has an entry in the dictionary
            if retailer not in retailer_product_dict:
                retailer_product_dict[retailer] = set()  # Use a set to store unique products

            # Add the product to the retailer's set of unique products
            retailer_product_dict[retailer].add(product)

        # Keep track of offers for each retailer
        if retailer not in retailer_offers:
            retailer_offers[retailer] = []

        # Add the offer to the retailer's list of offers
        retailer_offers[retailer].append(offer)

    # Filter retailers where all offers are NaN
    retailers_with_nan_offers = [retailer for retailer, offers in retailer_offers.items() if all(pd.isna(offer) for offer in offers)]




    # Define a function to get similar retailers and offers
    def get_similar_retailers_and_offers(retailer_name):
        # Check if the retailer is in retailer_product_dict
        if retailer_name in retailer_product_dict:
            # If the retailer has non-NaN offers, return them
            retailer_offers = ordered_df[ordered_df['RETAILER'] == retailer_name]['OFFER'].unique()
            return retailer_offers  # Just return the actual offers

        # Retailer not found in retailer_product_dict, find similar retailers
        similarity_scores = {}

        # Get the set of products associated with the retailer (including NaN offers)
        products_for_retailer = set(ordered_df[ordered_df['RETAILER'] == retailer_name]['Product'].unique())

        for other_retailer, other_products in retailer_product_dict.items():
            # Skip the same retailer
            if other_retailer == retailer_name:
                continue

            # Calculate Jaccard similarity between the two retailers
            intersection = len(products_for_retailer.intersection(other_products))
            union = len(products_for_retailer.union(other_products))

            # Handle cases where both retailers have no associated products (all NaN offers)
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union

            similarity_scores[other_retailer] = similarity

        # Sort similar retailers by similarity score in descending order
        sorted_similar_retailers = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out retailers with a similarity score of 0
        similar_retailer_offers = [(ordered_df[ordered_df['RETAILER'] == similar_retailer]['OFFER'].unique()) for similar_retailer, similarity_score in sorted_similar_retailers if similarity_score != 0]

        # If there are similar retailer offers, return them
        if similar_retailer_offers:
            return similar_retailer_offers

        # If there are no similar retailer offers, return "No offers"
        return ['No offers']


    # Replace 'Your Retailer Name' with the actual retailer name you want to find similar retailers for
    retailer_name = merchant
    similar_offers = get_similar_retailers_and_offers(retailer_name)
    
    return similar_offers


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_retail_offers', methods = ['POST', 'GET'])
def get_retail_offers():
    try:
        merchant = request.form['merchant']
        print(f'merchant is {merchant}')
        result = merchant_offers(merchant)
        print(f'offers is {result}')
        return render_template('page1.html', result=[merchant, result], my_string=merchant)
    except Exception as e:
        return render_template('page1.html', result=None, error=str(e))

@app.route('/get_cat_offers', methods = ['POST', 'GET'])
def get_cat_offers():
    try:
        category = request.form['category']
        print(f'category is {category}')
        result = fetch_category(category)
        print(f'offers is {result}')        
        return render_template('page2.html', result=[category, result], my_string=category)
    except Exception as e:
        return render_template('page2.html', result=None, error=str(e))

@app.route('/get_brand_offers', methods = ['POST', 'GET'])
def get_brand_offers():
    try:
        brand_ = request.form['brand']
        print(f'category is {brand_}')
        result = fetch_brand(brand_)
        print(f'offers is {result}')
        return render_template('page3.html', result=[brand_, result], my_string=brand_)
    except Exception as e:
        return render_template('page3.html', result=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
