import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax
import eclat

# Eclat Algorithm Source
# https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining

# Type 1 Data
# https://www.kaggle.com/mariekaram/apriori-association-rule
# https://www.kaggle.com/ekrembayar/apriori-association-rules-grocery-store/notebook

# Type 2 Data
# https://www.kaggle.com/nandinibagga/apriori-algorithm

ALLOWED_EXTENSIONS = ['.xlsx', '.xlx', '.csv', '.xml']
app = Flask(__name__)

uploads_dir = os.path.join(app.root_path, 'files_repository')
app.config['UPLOAD_EXTENSIONS'] = ALLOWED_EXTENSIONS

def preproces_dataset(path_filename):
  dataframe = pd.read_csv(path_filename)
  dataframe.rename(columns={'Member_number': 'customer_id', 'itemDescription': 'item'},inplace=True)
  dataframe.drop(columns=['Date'], inplace=True)
  #print("Items count before deleting duplicates: ", dataframe.shape[0])
  dataframe.drop_duplicates(inplace=True)
  #print("Items count after deleting duplicates: ", dataframe.shape[0])
  items_set = dataframe.groupby(by = ['customer_id'])['item'].apply(list).reset_index()
  items_list = items_set['item'].tolist()
  te = TransactionEncoder()
  te_ary = te.fit_transform(items_list)
  items_df = pd.DataFrame(te_ary, columns=te.columns_)
  return items_df

@app.route('/')
def home():
  return render_template('FrequentItemsetMining.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    # Get File and save
    f = request.files['file']
    filename = secure_filename(f.filename)
    if filename != '':
      file_ext = os.path.splitext(filename)[1]
      if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        Flask.abort(404, description="Resource not found")
        return Flask.redirect("/")
      else:
        f.save(os.path.join(uploads_dir, filename))

    # Get the fullpath of filename
    fullpath_filename = os.path.join(uploads_dir, filename)

    # Get Value and change the data type
    checkbox = request.form.getlist('algorithm') # List
    generateRules = request.form.getlist('flexSwitchCheck') # List
    minSup = float(request.form.get('rangeInput1')) # Float
    minThres = float(request.form.get('rangeInput2')) # Float
    maxItemsets = int(request.form.get('floatingSelect')) # Integer
    
    # Check Only
    print('Generate Rules: {} | MinSup: {} | MaxItemsets: {} | MinThres: {}'.format(generateRules, minSup, maxItemsets, minThres))
    print('{} | {} | {} | {}'.format(type(generateRules), type(minSup), type(maxItemsets), type(minThres)))
    print(filename)
    print(checkbox)

    # preprocess dataset
    items_df = preproces_dataset(fullpath_filename)

    # Prepare variable
    apriorirecords = None
    aprioricolnames = None

    # assign data
    if '1' in checkbox:
      aprioridf = apriori(items_df, min_support=minSup, use_colnames=True, max_len=maxItemsets)
      if 'True' in generateRules:
        rules = association_rules(aprioridf, min_threshold=minThres)
        rules.drop(rules.columns[[2, 3]], axis = 1, inplace = True)
        rules["support"] = rules["support"].apply(lambda x: format(float(x),".5f"))
        rules["lift"] = rules["lift"].apply(lambda x: format(float(x),".5f"))
        rules["leverage"] = rules["leverage"].apply(lambda x: format(float(x),".5f"))
        rules["conviction"] = rules["conviction"].apply(lambda x: format(float(x),".5f"))
        apriorirecords = rules.to_dict('records')
        aprioricolnames = rules.columns.values
      else:
        aprioridf['length'] = aprioridf['itemsets'].apply(lambda x: len(x))
        apriorirecords = aprioridf.to_dict('records')
        aprioricolnames = aprioridf.columns.values
      del aprioridf

    del items_df
    return render_template('MiningAnalysis.html', algorithms=checkbox,
      apriorirecords=apriorirecords, aprioricolnames=aprioricolnames
      )

if __name__ == '__main__':
  app.run(debug = True)