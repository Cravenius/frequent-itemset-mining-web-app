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

    # Preprocess File

    # Get Value
    generateRules = request.form.getlist('flexSwitchCheck')
    checkbox = request.form.getlist('algorithm')
    minSup = request.form.get('rangeInput1')
    minThres = request.form.get('rangeInput2')
    maxItemsets = request.form.get('floatingSelect')
    
    # Check Only
    print('Generate Rules: {} | MinSup: {} | MaxItemsets: {} | MinThres: {}'.format(generateRules, minSup, maxItemsets, minThres))
    print(filename)
    print(checkbox)


    # Read File
    df = pd.read_csv(os.path.join(uploads_dir, filename))


    # Before submit
    temp = df.to_dict('records')
    columnNames = df.columns.values

    return render_template('MiningAnalysis.html', records=temp, colnames=columnNames)

if __name__ == '__main__':
  app.run(debug = True)