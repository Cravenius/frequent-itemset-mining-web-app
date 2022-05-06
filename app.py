# https://www.kaggle.com/mariekaram/apriori-association-rule
# https://www.kaggle.com/ekrembayar/apriori-association-rules-grocery-store/notebook
# https://www.kaggle.com/nandinibagga/apriori-algorithm

import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, flash, request, redirect, url_for, send_from_directory
from sklearn import metrics
from werkzeug.utils import secure_filename
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax
import seaborn as sns
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.figure import Figure
from pandas.plotting import parallel_coordinates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

plt.switch_backend('agg')
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

def rules_to_coordinates(rules):
  rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
  rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
  rules['rule'] = rules.index
  return rules[['antecedent','consequent','rule']]

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
    filter = request.form.getlist('floatingSelectMetric')[0] # List
    
    # Check Only
    print(filter)
    print('MinSup: {} | MaxItemsets: {} | MinThres: {}'.format(minSup, maxItemsets, minThres))
    print('{} | {} | {} | {}'.format(type(generateRules), type(minSup), type(maxItemsets), type(minThres)))
    print(filename)
    print(checkbox)

    # preprocess dataset
    items_df = preproces_dataset(fullpath_filename)

    # Prepare variable
    fpgrowthrecords = None
    fpgrowthcolnames = None
    rulesrecords = None
    rulescolnames = None
    heatmap_plot = []

    # assign data
    if '1' in checkbox:
      fpgrowthdf = fpgrowth(items_df, min_support=minSup, use_colnames=True, max_len=maxItemsets)
      rules = association_rules(fpgrowthdf, min_threshold=minThres, metric=filter)
      rules.drop(rules.columns[[2, 3]], axis = 1, inplace = True)
      rules["support"] = rules["support"].apply(lambda x: format(float(x),".5f"))
      rules["confidence"] = rules["confidence"].apply(lambda x: format(float(x),".5f"))
      rules["lift"] = rules["lift"].apply(lambda x: format(float(x),".5f"))
      rules["leverage"] = rules["leverage"].apply(lambda x: format(float(x),".5f"))
      rules["conviction"] = rules["conviction"].apply(lambda x: format(float(x),".5f"))
      rulesrecords = rules.to_dict('records')
      rulescolnames = rules.columns.values

      fpgrowthdf['length'] = fpgrowthdf['itemsets'].apply(lambda x: len(x))
      fpgrowthrecords = fpgrowthdf.to_dict('records')
      fpgrowthcolnames = fpgrowthdf.columns.values

      # Scatter Plot
      fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
      sns.scatterplot(x = "support", y = "confidence", data = rules)
      plt.margins(0.01, 0.01)
      plt.xticks(rotation='vertical')
      fig.savefig('static/scatter_plot.png')   # save the figure to file
      plt.close(fig)

      fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
      sns.scatterplot(x = "support", y = "confidence", size = filter, data = rules)
      plt.margins(0.01, 0.01)
      plt.xticks(rotation='vertical')
      fig.savefig('static/scatter_plot_optimal.png')   # save the figure to file
      plt.close(fig)

      # Parallel Coordinates Plot
      fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
      coords = rules_to_coordinates(rules)
      parallel_coordinates(coords, 'rule')
      plt.legend([])
      plt.grid(True)
      fig.savefig('static/parallel_coordinates.png')   # save the figure to file
      plt.close(fig)

      # Convert antecedents and consequents into strings
      rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
      rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

      fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
      support_table = rules.pivot(index='consequents', columns='antecedents', values='support')
      support_table = support_table.fillna(0)
      for col in support_table.columns:
        support_table[col] = pd.to_numeric(support_table[col],errors = 'coerce')
      print(support_table.info())

      sns.heatmap(support_table, annot=True, cbar=False, ax=ax)
      b, t = plt.ylim() 
      b += 0.5 
      t -= 0.5 
      plt.ylim(b, t) 
      plt.yticks(rotation=0)
      plt.margins(0.01, 0.01)
      fig.savefig('static/heatmap.png')   # save the figure to file
      plt.close(fig)

      fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
      support_table = rules.pivot(index='consequents', columns='antecedents', values=filter)
      support_table = support_table.fillna(0)
      for col in support_table.columns:
        support_table[col] = pd.to_numeric(support_table[col],errors = 'coerce')
      print(support_table.info())
      
      sns.heatmap(support_table, annot=True, cbar=False, ax=ax)
      b, t = plt.ylim() 
      b += 0.5 
      t -= 0.5 
      plt.ylim(b, t) 
      plt.yticks(rotation=0)
      plt.margins(0.01, 0.01)
      fig.savefig('static/heatmap_optimal.png')   # save the figure to file
      plt.close(fig)

    del items_df
    del rules
    del fpgrowthdf
    return render_template('MiningAnalysis.html', algorithms=checkbox,
      fpgrowthrecords=fpgrowthrecords, fpgrowthcolnames=fpgrowthcolnames,
      rulesrecords=rulesrecords, rulescolnames=rulescolnames
      )

if __name__ == '__main__':
  app.run(debug = True)