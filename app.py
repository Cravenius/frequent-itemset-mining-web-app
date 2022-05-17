# https://www.kaggle.com/mariekaram/apriori-association-rule
# https://www.kaggle.com/ekrembayar/apriori-association-rules-grocery-store/notebook
# https://www.kaggle.com/nandinibagga/apriori-algorithm

import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth
import base64
from io import BytesIO
import plotly
import plotly.express as px
import json

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

def plotly_heatmap(rules, filter, color, title):
  st = rules.pivot(index='consequents', columns='antecedents', values=filter)
  st = st.fillna(0)
  for col in st.columns:
    st[col] = pd.to_numeric(st[col],errors = 'coerce')
  print(st.info())
  fig = px.imshow(st, labels=dict(x="Item", y="Item", color="Filter: {}".format(filter), title=title),
    x=st.columns.tolist(),
    y=st.index.tolist(),
    aspect="auto", text_auto=True,
    color_continuous_scale=color,
    )
  fig.update_xaxes(side="top")
  fig.update_layout(width=1250, height=600, margin=dict(l=20, r=20, t=20, b=20))
  fig.layout.autosize = True
  plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return plot_json

def plotly_scatter(df, filter, title):
  fig = px.scatter(df, x="support", y="confidence", color=filter, marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white", title=title)
  fig.update_xaxes(side="top")
  fig.update_layout(width=1250, height=800)
  plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return plot_json

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
    scatter_support = None
    scatter_filter = None
    heatmap_support = None
    heatmap_filter = None

    # assign data
    if '1' in checkbox:
      fpgrowthdf = fpgrowth(items_df, min_support=minSup, use_colnames=True, max_len=maxItemsets)
      fpgrowthdf['length'] = fpgrowthdf['itemsets'].apply(lambda x: len(x))
      fpgrowthrecords = fpgrowthdf.to_dict('records')
      fpgrowthcolnames = fpgrowthdf.columns.values
      
      rules = association_rules(fpgrowthdf, min_threshold=minThres, metric=filter)
      rules.drop(rules.columns[[2, 3]], axis = 1, inplace = True)
      rules["support"] = rules["support"].apply(lambda x: format(float(x),".5f"))
      rules["confidence"] = rules["confidence"].apply(lambda x: format(float(x),".5f"))
      rules["lift"] = rules["lift"].apply(lambda x: format(float(x),".5f"))
      rules["leverage"] = rules["leverage"].apply(lambda x: format(float(x),".5f"))
      rules["conviction"] = rules["conviction"].apply(lambda x: format(float(x),".5f"))
      rulesrecords = rules.to_dict('records')
      rulescolnames = rules.columns.values

      scatter_support = plotly_scatter(rules, 'support', 'scatter plot with support filter')
      scatter_filter = plotly_scatter(rules, filter, 'scatter plot with {} filter'.format(filter))

      # Convert antecedents and consequents into strings
      rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
      rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
   
      heatmap_support = plotly_heatmap(rules, 'support', 'RdPu', 'Support Correlation Table')
      heatmap_filter = plotly_heatmap(rules, filter, 'GnBu', "{} Correlation Table".format(filter))

    del items_df
    del rules
    del fpgrowthdf
    return render_template('MiningAnalysis.html', algorithms=checkbox,
      fpgrowthrecords=fpgrowthrecords, fpgrowthcolnames=fpgrowthcolnames,
      rulesrecords=rulesrecords, rulescolnames=rulescolnames,
      heatmap_support=heatmap_support, heatmap_filter=heatmap_filter, 
      scatter_support=scatter_support, scatter_filter=scatter_filter
      )

if __name__ == '__main__':
  app.run(debug = True)