import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

def split_vessel(vessel):
    x = vessel.split(" ")
    if len(x) == 1:
        return vessel
    if  x[-1]=="Vessel":
        x[-1]=""
    if x[0]=="Daily" or x[0]=="Motor" or x[0]=="Floating":
        x[0]=""
    if x[1]=="Boat" or x[1]=="War":
        x[1]=""
    if(x[-1]=="Yacht"):
        vessel= "Yacht"
        return vessel
    if x[0]=="Barges":
        x[0]="Other"
    return ''.join(x)

def Donut(filename,total_cost):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df.Date, unit=None)
    df['Date'] = df['Date'].dt.month
    # df['Delay'] = df['Delay'].apply( lambda x: x.split('days ')[1])

    df['Delay'] = df['Delay'].apply( lambda x: datetime.strptime(x, "%H:%M:%S"))

    df['Delay'] = df['Delay'].apply( lambda x: x.hour*60 + x.minute + x.second/60)

    ddf=df

    # ddf['Date']=ddf['Date'].astype(int)
    # ddf['Date'] = ddf['Date'].apply( lambda x: calendar.month_name[x])
    # df2=ddf.groupby(['Date'])

    color_list = ['#2f6694','#b7d2e8', '#caddee', '#679fce', '#1d3f5c', '#b7d2e8', '#193750', '#8a71cc', '#856bca','#ded7f1','#8165c8','#1c1336']

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height


    ddf['Vessel Type']= ddf['Vessel Type'].apply( lambda x: split_vessel(x))
    df2=ddf.groupby(['Vessel Type'])['Delay'].sum().to_frame().reset_index(drop=False)
    df2['Delay'] = df2['Delay'].apply(pd.to_numeric)
    df2.loc[df2['Delay'] < df2['Delay'].quantile(.75),'Vessel Type']="Other"
    df2=df2.groupby(['Vessel Type'])['Delay'].sum().to_frame().reset_index(drop=False)
    
    #Vessel_Types= list(df2['Vessel Type'])
    Vessel_Types = [df2['Vessel Type'][i]
    for i in range(len(df2['Vessel Type']))
        if df2['Delay'][i] > 0]
    
    df2= df2.loc[df2['Delay'] >= 0]
    
    if (df2['Delay'].sum()== 0):
        
        df2['Vessel Type']= df2['Vessel Type'].apply( lambda x: "Other")
        
        df2=df2.groupby(['Vessel Type'])['Delay'].sum().to_frame().reset_index(drop=False)
        
        df2['Delay'] = df2['Delay'].replace([0], 1)
                      
        fig=  plt.figure(figsize=(6,4),facecolor='#222')
        txt="Total Delay "+'\n'+total_cost
        
        df_to_list = df2['Delay'].to_list()
        plt.pie(df_to_list,colors=color_list)
            
        fig.text(0.5 * (left + right), 0.5 * (bottom + top), txt,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize = 15, weight='bold')
        
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.tight_layout()      
    
    else:
        fig =  plt.figure(figsize=(6,4),facecolor='#222')
        df_to_list = df2['Delay'].to_list()
        plt.pie(df_to_list,colors=color_list)
        plt.legend(Vessel_Types,ncols=3,loc="lower center",fontsize="10")
        plt.axis('off')
        
        txt="Total Delay "+'\n'+total_cost
        
        fig.text(0.5 * (left + right), 0.5 * (bottom + top), txt,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize = 15, weight='bold')
        
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.tight_layout()


    plt.savefig(f'Vessel_Type_Donut.png', dpi=300, bbox_inches='tight')

    #plt.show()