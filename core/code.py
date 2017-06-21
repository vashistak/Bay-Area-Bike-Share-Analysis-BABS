
import os
os.chdir("D:\\flask-app-master")
os.getcwd()
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from flask import send_file
import json

#%matplotlib inline
import plotly
plotly.offline.init_notebook_mode() # run at the start of every notebook
import pandas as pd
import datetime as dt
from matplotlib.pyplot import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from io import BytesIO
def fulldata():
	colnames = ['id', 'duration', 's_date', 'start_station',
                 's_terminal', 'e_date', 'end_station', 'e_terminal',
                 'bike', 'type', 'zip']

        #changing column names for ease of data
	#reading full raw data
	full_data = pd.read_csv(".\GDADATA\GDA_DATA.csv", names=colnames,skiprows=[0],low_memory= False,skipinitialspace=True)

	full_data['duration'] = full_data['duration'].apply(lambda x: x/60)
	full_data.duration = full_data.duration.round()
	return full_data

def prepare_data(data):
    '''
        Data Quality check and improvement
    '''
    
    #Changing duration of rides into minutes
    #data['duration'] = data["duration"]/60
    #data.loc[:,"duration"]/=60
    #data['duration'] = data['duration'].apply(lambda x: x/60)
       
     #splitting date and time  
    data['s_date'] =  pd.to_datetime(data['s_date'], format='%m/%d/%Y %H:%M')
    data['start_date'] = [d.date() for d in data['s_date']]
    data['start_time'] = [d.time() for d in data['s_date']]

    data['e_date'] =  pd.to_datetime(data['e_date'], format='%m/%d/%Y %H:%M')
    data['end_date'] = [d.date() for d in data['e_date']]
    data['end_time'] = [d.time() for d in data['e_date']]

    #splitting hour of day
    data['start_hour'] = data.start_time.apply(lambda x: x.hour)
    data['end_hour'] = data.end_time.apply(lambda x: x.hour)
    
    #splitting year and month
    data['year'], data['month'] = data['start_date'].apply(lambda x: x.year), data['end_date'].apply(lambda x: x.month)

    #dropping prior column of datetime
    data = data.drop(['s_date','e_date'], axis=1)
    
    #with weekday identifying weekdays and weekends of the week
    data['day'] = data['start_date'].apply(lambda x: x.weekday())
    '''
    data['day'].replace([0,1,2,3,4], 'Weekday',inplace=True)
    data['day'].replace([5,6], 'Weekend',inplace=True)
    '''
    data['day'].replace(0, 'Monday',inplace=True)
    data['day'].replace(1, 'Tuesday',inplace=True)
    data['day'].replace(2, 'Wednesday',inplace=True)
    data['day'].replace(3, 'Thursday',inplace=True)
    data['day'].replace(4, 'Friday',inplace=True)
    data['day'].replace(5, 'Saturday',inplace=True)
    data['day'].replace(6, 'Sunday',inplace=True)
    
    #changing month number into month name
    import calendar
    data['month'] = data['month'].apply(lambda x: calendar.month_abbr[x])
    
    #making hour, month, year, day as categorical values
    data[['start_hour', 'end_hour','year','month','day']].apply(lambda x: x.astype('category'))
    
    #cutting duration of rides into buckets of minutes 
    def duration_bucket(a):
        if a <= 5: return '<5'
        elif 5 < a <= 10 : return '5-10 min'
        elif 10 < a <= 15: return '10-15 min'
        elif 15 < a <= 20: return '15-20 min'
        elif 20 < a <= 25 : return '20-25 min'
        elif 25 < a <= 30: return '25-30 min'
        elif 30 < a <= 35: return '30-35 min'
        elif 35 < a <= 40 : return '35-40 min'
        elif 40 < a <= 45: return '40-45 min'
        elif 45 < a <= 50: return '45-50 min'
        elif 50 < a <= 55 : return '50-55 min'
        elif 55 < a <= 60: return '55-60 min'
        elif 60 < a <= 180: return '1-3 hour'
        elif 180 < a <= 360 : return '3-6 hour'
        elif 360 < a <= 540: return '6-9 hour'
        elif 540 < a <= 720: return '9-12 hour'
        elif 12 < a <= 24 : return '12-24 hour'
        else: return '> 1day'
    
    data['d_bucket'] = data['duration'].apply(lambda c: duration_bucket(c))
    
    return data
	

def data_make(start,dest):

    '''
        data_make prepares data based on starting station and destination given.
        If all is selected full data is prepared after data quality checking
    '''
    full_data=fulldata()

    if start=="All" and dest == "All":
        filter_trip=full_data
    elif start=="All":
        filter_trip = full_data[full_data['end_station'] == dest].copy()
    elif dest=="All":
        filter_trip = full_data[full_data['end_station'] == start].copy()
    else:
        filter_trip = full_data[(full_data['start_station'] == start) & (full_data['end_station'] == dest)].copy()
    df=prepare_data(filter_trip)
    return df


def dataforpie(df):

    '''
        Data frame passed is converted into json format for chart.js returned into
        getting_form2.
    '''
    gdata = {}
	
    gdata["labels"] = df['type'].tolist()
    data = {}
    data["data"] = df['sub_Count'].tolist()
    x = []
    
    gdata["datasets"] = x

    bgColor = []
    data['backgroundColor'] = bgColor
    a = ['#3e95cd', '#8e5ea2']
    for x in a:
            data['backgroundColor'].append(x) 
    gdata["datasets"].append(data)

    
    return json.dumps(gdata)

def dataforplotly2(data,secondary_group,analyse_by,data_col,primary_group='type'):
    data_sub=data[data[primary_group] == 'Subscriber' ],
    data_sub_1=data[data[primary_group] == 'Customer' ]
    
    graphs=[{
           "data": [
                    {
                      "y": data[data[primary_group] == 'Subscriber' ][data_col].tolist(),
                      "x": data[data[primary_group] == 'Subscriber' ][secondary_group].tolist(),
                      "type": "bar",
                      "name": "Subscriber"
                    },
                    {
                      "y": data[data[primary_group] == 'Customer' ][data_col].tolist(),
                      "x": data[data[primary_group] == 'Customer' ][secondary_group].tolist(),
                      "type": "bar",
                      "name": "Customer"
                    }
                  ],
                  "layout":{
                    "barmode": "group",
                    "title": 'Analysis for %s by '%secondary_group+'Total %s'%analyse_by+' of Rides',
                     
                  }
                }]
        
    return graphs

        
def dataforplot(data, secondary_group,analyse_by, top,primary_group):
    '''
       param:data = Data frame
       param: secondary group : "column name" ("start station","end station","time","month","day")
       param analyse_by: total number or total durationof rides
       param: primary group "type" Customer or Subscriber

       Data is filtereed and grouped based on secondary and primary group provided
       if top values are selected then data is sorted for major group required.
       returns: data, data_col: grouped data, data_col for analyse by
    '''
        
    
    if secondary_group == 'month' or secondary_group == 'day' or secondary_group == 'd_bucket':
        top = False
    # grouping rides primary and secondary group of given columns
    data_grouped = data[['duration', primary_group, secondary_group]] \
                        .groupby([primary_group, secondary_group]) \
                        .agg({"duration": {'Total Duration': 'sum', 
                                  "Total Rides": 'count'}}).reset_index()
    

    data_grouped.columns = [''.join(col).strip().replace('duration', '') for col in data_grouped.columns.values]
    
    if analyse_by=='Number':
    #Filtering and sorting top secondary group passed #Top start or destination station   
    
        tops = data[['duration', secondary_group]].groupby(secondary_group).count().reset_index()

    #selecting only top values for given passed int else no filter applied.
    #no filter applied while stuying graphs for week, month, time wise analysis
        
        data_col="Total Rides"
    else:
        tops = data[['duration', secondary_group]].groupby(secondary_group).sum().reset_index()
        data_col="Total Duration"

        
    
    tops = tops.sort_values(by='duration', ascending=False) \
               .iloc[:min(top, tops.shape[0]) if top else None]
    tops = list(tops[secondary_group].unique())
        
    data_grouped = data_grouped.sort_values(by=[data_col,primary_group,secondary_group], ascending=False)
    data_grouped.index = [str(x) for x in data_grouped.index]
    
    #selecting only rows present in top filtered data
    data_grouped = data_grouped[data_grouped[secondary_group].isin(tops)]
    #data_grouped = data_grouped.pivot(index=secondary_group, columns=primary_group,
                #      values=data_col).reset_index()
    return data_grouped,data_col
def data_new(data,colum):
    '''
       param:data = Data frame
       param: column group "Total_Duration" or "Number_of_Rides" 

       Data is filtered and grouped based on colum value provided
       returns: data, data_col: grouped data, data_col for analyse by
    '''
    data_grouped = data[['duration', 'type', 'start_date']] \
                        .groupby(['type', 'start_date']) \
                        .agg({"duration": {'Total_Duration': 'sum', 
                                  'Number_of_Rides': 'count'}}) \
                        .reset_index()

    data_grouped.columns = [''.join(col).strip().replace('duration', '') for col in
                              data_grouped.columns.values]

    #print(data_grouped.head())
    cols = colum
    return data_grouped,cols
def dataforplotly3(data,col,s_station,e_station):
    
    data_group = data.pivot('start_date', 'type', col)
    data_group=data_group.reset_index()
    
   
    #print(data_group)
 
    graphs= [{
              "data": [
                         {
      
                              "y": (data_group['Customer']+data_group['Subscriber']).tolist(),
                              "x": data_group['start_date'].tolist(),
                              "fill": "tonexty",
                              "type": "scatter",
                              "name": "Total"
                             },
                        {
      
                              "y": data_group['Customer'].tolist(),
                              "x": data_group['start_date'].tolist(),
                              "fill": "tonexty",
                              "type": "scatter",
                              "name": "Customer"
                             },
                        {
      
                              "y": data_group['Subscriber'].tolist(),
                              "x":data_group['start_date'].tolist(),
                              "fill": "tonexty",
                              "type": "scatter",
                              "name": "Subscriber"
                             }
                     ],
                     "layout": {
                                "boxmode": "overlay",
                                "title": 'Trend for'+' %s '%col +'between' +' %s '%s_station+' stations to'+ ' %s '%e_station + 'Destinations'
                                
                                }
        
                    }]
    return graphs

'''
Use only when plotly does not installs. 
def makeaplot(data,secondary_group,primary_group,analyse_by):
    data.dropna(thresh=2)
    data.fillna(0,inplace=True)
  
    data = data.sort_values(by=['Subscriber'],ascending=False)
    
    data_cus = np.array(data['Customer'])
    data_sub = np.array(data['Subscriber'])
    fig, ax = plt.subplots(figsize=(15,10))
    corner = np.arange(data.shape[0])
    rect1=ax.bar(corner, data_cus, width=0.2, label='Customer',align='center')
    rect2=ax.bar(corner+0.3, data_sub,width=0.2,  color='b', label='Subscriber',align='center')
    def autolabel(rects):
        for rect in rects:
                h = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                                ha='center', va='bottom')
    autolabel(rect1)
    autolabel(rect2)
    ax.set_xlabel(secondary_group,fontsize=12)
    ax.set_ylabel('Total %s of Rides'%analyse_by,fontsize=12)
    plt.legend(loc='best')
    plt.title('Analysis for %s by '%secondary_group+'Total %s'%analyse_by+' of Rides',fontsize=12 )
    fig.subplots_adjust(bottom=0.28)
    plt.tight_layout(pad=3)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xticks(corner+0.125,data[secondary_group],rotation=90, fontsize=10)
    canvas=FigureCanvas(fig)
    img= BytesIO()
    fig.savefig(img,bbox_inches='tight')
    img.seek(0)
    return send_file(img, mimetype='image/png')
'''
    
