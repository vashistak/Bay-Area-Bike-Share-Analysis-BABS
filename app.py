from flask import Flask,flash,abort, current_app, jsonify, Response,session, render_template, request, redirect,send_file, make_response
import requests
import plotly
import pandas as pd

import json

import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
	return redirect('/index')

#loading form.html file for reading data and canvas for pie chart
@app.route('/index')
def index():
    return render_template('form.html', message = "")

#extracting functions from code file to app

from core.code import data_new,dataforpie,dataforplotly3,dataforplot,dataforplotly2,data_make,prepare_data,fulldata


'''
#getting_form2 executes on canvas created in form =.html
#getting_form2 uses html chart.js to generate pie chart. ajax json data
#into html to draw pie chart.
'''

@app.route('/getting_form2',methods=['POST'])

def getting_form2():

    #print(request.data)

    input = request.data
    inStrs = input.decode('utf-8').split('|')

    #print(inStrs)

    # Getting the data from the form
    s_station = str(inStrs[0])
    e_station = str(inStrs[1])
    data=data_make(s_station,e_station)
    df = pd.DataFrame({'sub_Count': data.groupby(['type']).size()})
    df=df.apply(lambda c: np.round(c*100/c.sum(),2),axis=0).reset_index()
    
    graphData=dataforpie(df)
    
    return Response(graphData, mimetype='application/json')
'''
#getting_form,getting_form3 out put files are in layouts/index.html
#filtered data is turned into plotly data structure json format and
#returned to getting_form then plotlyjsonencoder dumps json and renders
#to index.html where plotly plots interactive chart
'''

@app.route('/getting_form', methods=['POST', 'GET'])
def getting_form():
    '''    
    Filtered data by routes, number of rides or duration of rides, top
    values if major data is selected.
    Getting the data from the form page
    '''
    s_station = str(request.form['Start_Station'])
    e_station = str(request.form['Destination'])
    data=data_make(s_station,e_station)
    
    top = int(request.form['top'])
    volumeorsum = request.form['Volume_Sum']
   
    s_group = request.form['secondaryBy']
    
    data_grouped,data_col = dataforplot(data, s_group,volumeorsum,top,primary_group='type')
    
    
    
    val= dataforplotly2(data_grouped,s_group,volumeorsum,data_col,primary_group='type')
    
    ids = ['graph-{}'.format(i) for i, _ in enumerate(val)]
    
    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    
    graphJSON1 = json.dumps(val, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('layouts/index.html',
                           ids=ids,
                           graphJSON=graphJSON1)

'''
#filtered data is turned into plotly data structure json format and
#returned to getting_form then plotlyjsonencoder dumps json and renders
#to index.html where plotly plots interactive chart
'''

@app.route('/getting_form3',methods=['POST','GET'])

def getting_form3():
        
    #Filtering data by routes for trend analysis. 3 tab in tool.
    
    s_station = str(request.form['Start_Station'])
    e_station = str(request.form['Destination'])
    month =str(request.form['Month'])
    colum  = str(request.form['by'])

    data=data_make(s_station,e_station)
    if month=="All":
        df=data
    else:
        df=data[data['month']==month]
    data_got,cols=data_new(df,colum)
    
    graphs=dataforplotly3(data_got,cols,s_station,e_station)
    

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    
    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('layouts/index.html',
                           ids=ids,
                           graphJSON=graphJSON)
if __name__ == '__main__':
	app.run(port=4762, debug=True)

'''
Choose for matplotlib plots if plotly does not install
@app.route('/image/')
def images():
    return render_template("image.html")
'''
