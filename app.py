import flask
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax
import eclat

# https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining

#ALLOWED_EXTENSIONS = {'xlsx', 'csv', 'xml'}
app = Flask(__name__)

uploads_dir = os.path.join(app.root_path, 'files_repository')

@app.route('/')
def home():
  return render_template('FrequentItemsetMining.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':

    #checkbox = request.form.getlist('algorithm')
    #if ('1' not in checkbox) or (1 not in checkbox):
      #return flask.redirect('/')

    f = request.files['file']
    f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
    df = pd.read_csv(os.path.join(uploads_dir, f.filename))
    temp = df.to_dict('records')
    columnNames = df.columns.values
    minSup = request.form.get('rangeInput')
    maxItemsets = request.form.get('floatingSelect')

    #print('MinSup: {} | MaxItemsets: {}\n'.format(minSup, maxItemsets))

    return render_template('output.html', records=temp, colnames=columnNames)

if __name__ == '__main__':
  app.run(debug = True)