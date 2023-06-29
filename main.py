import pandas as pd
import dash
import proto
from dash import dash_table
from dash import dcc
from dash import html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import tqdm
import datetime
import requests
from bs4 import BeautifulSoup
import math
import base64
import dash_auth
import config
import os.path
import time
from datetime import timedelta

my_results = ''
max_disc = 20
refresh_interval = 4
early_stopping = True
tol = 1 ** (math.e - 4)
pix = 50
max_r = 500
my_val = 20
image_path = 'start.png'
encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')

final_df = pd.DataFrame() #crate final empty data frame

temp1 = pd.read_excel(config.data_file, parse_dates=['ΗΜΕΡ/ΝΙΑ ΑΦΙΞΗΣ']) #data for scheduler
temp1.rename(columns={'ΟΝΟΜΑΣΙΑ ΠΛΟΙΟΥ': 'VESSEL_NAME'}, inplace=True)

temp2 = pd.read_excel(config.data_file2) #data with IMO
temp2.rename(columns={'IMO/ΑΡ.ΝΗΟΛΟΓΙΟΥ': 'IMO'}, inplace=True)
temp2.rename(columns={'ΟΝΟΜΑΣΙΑ ΠΛΟΙΟΥ': 'VESSEL_NAME'}, inplace=True)


temp3 = temp2.loc[:, ["IMO","VESSEL_NAME","Κ.Κ.Χ."]].copy()
temp3 = temp3.drop_duplicates()
my_df = pd.merge(temp1, temp3, on=['VESSEL_NAME', 'Κ.Κ.Χ.'], how='left')

my_df = my_df.round(decimals=5)
#my_df['IMO'] = my_df['IMO'].astype('string')
my_df.rename(columns={'ΗΜΕΡ/ΝΙΑ ΑΦΙΞΗΣ': 'TIMESTAMP'}, inplace=True)
my_df.rename(columns={'ΚΑΤΗΓ.ΤΙΜ/ΓΗΣΗΣ': 'Corinth Canal Invoice Category'}, inplace=True)
my_df.rename(columns={'Κ.Κ.Χ.': 'NET_TONNAGE'}, inplace=True)

mt_df = pd.read_excel("Survey Marine Traffic Data_19_20_ARXIMIDIS.xlsx")
mt_df = mt_df.round(decimals=5)
mt_df['IMO'] = mt_df['IMO'].astype('string')

df_MGO = pd.read_excel('Bunker_Index_VLSFO_IMO2020_Arximidis.xlsx', 1) #data for MGO
df_MGO.columns = df_MGO.iloc[5]
df_MGO = df_MGO.iloc[6:]
df_MGO.reset_index(drop=True, inplace=True)
df_MGO.drop(df_MGO.columns[[1, 2, 3]], axis=1, inplace=True)

df_MGO.columns = ['TIMESTAMP', 'MGO1', 'MGO2', 'MGO3', 'MGO4', 'MGO_P', 'MGO_I']
df_MGO.drop(['MGO1', 'MGO2', 'MGO3', 'MGO4'], axis=1, inplace=True)
df_MGO['Average'] = df_MGO[['MGO_P', 'MGO_I']].mean(axis=1)
df_MGO.to_csv('MGO.csv')
my_df['Average MGO'] = df_MGO['Average']


def get_sched():
    global my_results
    df_sched = pd.read_csv(my_results)
    df_sched.rename(columns={'ΩΡΑ ΕΙΣΟΔΟΥ': 'Begin'}, inplace=True)
    df_sched.rename(columns={'ΩΡΑ ΕΞΟΔΟΥ': 'End'}, inplace=True)
    df_sched.rename(columns={'ΚΑΘΥΣΤΕΡΗΣΗ': 'Delay'}, inplace=True)
    df_sched.rename(columns={'ΑΝΑΜΕΝΟΜΕΝΟΣ ΧΡΟΝΟΣ ΔΙΑΣΧΙΣΗΣ': 'Estimated time to cross'}, inplace=True)
    df_sched = df_sched.round(decimals=5)
    return df_sched



# df_bunker = pd.read_excel('Bunker_Index_VLSFO_IMO2020_Arximidis.xlsx')




# Formula for Peloponese Circle : ("Fuel Consumption Per Day (mtns)"/24)*MGO*"Saving Time (Service Speed)"

#MGO SCRAPER
def get_emea_bunker_prices(site='https://shipandbunker.com/prices/emea', verbose=False):
    soup, rows, ix = BeautifulSoup(requests.get(site).text), [], []

    for rec in tqdm.tqdm(soup.findAll("tr", {"class": "row"}), disable=not verbose):
        ix.append(rec.find("th", {"scope": "row"}).find("a").text)

        timestamp, bunker_indices = rec.find("td", {"class": "date"}), \
            [col.text for col in rec.findAll("span", {"class": "tSearch"})]

        rows.append(
            [timestamp if timestamp is None else f'{timestamp.text} {datetime.date.today().year}', *bunker_indices])

    emea_bunker_prices = pd.DataFrame(rows, columns=["Date", "VLSFO", "MGO", "IFO380"],
                                      index=pd.Index(ix, name='Port')).dropna()
    emea_bunker_prices.Date = pd.to_datetime(emea_bunker_prices.Date, format="%b %d %Y")

    return emea_bunker_prices


def get_all_data():
    all_data = my_df.sort_values('TIMESTAMP', ascending=False)
    return all_data


def get_data(num):
    data = get_all_data().iloc[num:max_r, :]
    return data


def merge_MGO():
    df_MGO['TIMESTAMP'] = df_MGO['TIMESTAMP'].astype('datetime64[ns]')
    t_df = my_df
    t_df['TIMESTAMP'] = t_df['TIMESTAMP'].astype('datetime64[ns]')
    temp = pd.merge(t_df, df_MGO, on=['TIMESTAMP'], how='left')
    temp = temp.drop_duplicates()
    return temp

def merge_MarineTraffic():
    global mt_df

    mt_temp = mt_df.loc[:, ["IMO", "Fuel Consumption \nPer Day (mtns)", "Saving Time (Service Speed)\n"]].copy()
    mt_temp = mt_temp.drop_duplicates()

    t_df = merge_MGO()

    temp = pd.merge(t_df, mt_temp, on=['IMO'], how='left')
  #  temp =
    return temp

def get_final_df():
    global final_df, max_disc

    max_discount = max_disc
    max_discount = -(1) * max_discount / 100

    final_df = merge_MarineTraffic()
    final_df.rename(columns={'TIMESTAMP': 'DATE'}, inplace=True)
    final_df.rename(columns={'ΚΑΤΕΥΘΥΝΣΗ': 'Dir'}, inplace=True)
    final_df.rename(columns={'ΑΡ.ΔΙΑΠΛΟΥ': 'Cross No'}, inplace=True)
    final_df['Begin'] = None
    final_df['End'] = None
    final_df['Delay'] = None
    final_df['ETC'] = None
    final_df['Cost vs Pel'] = None
    final_df['Discount'] = None

    final_df = final_df.loc[final_df['DATE']==pd.to_datetime(config.date)]
    final_df['DATE'] = pd.to_datetime(final_df["DATE"]).dt.date
    final_df = total_fees(final_df)

    final_df.rename(columns={'Towage Fees': 'Towage'}, inplace=True)
    final_df.rename(columns={'Total Transit Fees': 'Total Fees'}, inplace=True)
    final_df.rename(columns={'Tolls Fees': 'Tolls'}, inplace=True)
    final_df['Pel.Cost'] = (final_df['Fuel Consumption \nPer Day (mtns)']) / 24 * final_df['MGO_P'] * final_df[
        'Saving Time (Service Speed)\n']
    final_df['Pel.Cost'] = final_df['Pel.Cost'].astype('float').round(2)
    final_df['Cost vs Pel'] = final_df['Pel.Cost'] - final_df['Total Fees']
    final_df['Cost vs Pel'] = final_df['Cost vs Pel'].astype(float).round(2)

    final_df['Discount'] = (final_df['Pel.Cost'] / final_df['Total Fees']) - 1
    final_df.loc[final_df['Discount'] > 0, 'Discount'] = 0
    final_df.loc[final_df['Discount'] < max_discount, 'Discount'] = max_discount
    final_df.loc[final_df['Cost vs Pel'].isnull(), 'Discount'] = 0
    final_df.loc[final_df['Discount'] == 0, 'Discount'] = (-1)*(final_df.loc[final_df['Discount'] == 0, 'Total Fees'] - final_df.loc[final_df['Discount'] == 0, 'Total Fees'] * (1 + max_discount))
    final_df.loc[(final_df['Discount'] <= max_discount), 'Total Fees'] = final_df.loc[
        (final_df['Discount'] <= max_discount), 'Total Fees'] * (1 + max_discount)
    final_df.loc[final_df['Discount'] == 0, 'Total Fees'] = final_df.loc[final_df['Discount'] == 0, 'Total Fees'] * (1 + max_discount)
    final_df.loc[(final_df['Discount'] < 0) & (final_df['Discount'] > max_discount), 'Total Fees'] = final_df.loc[
        (final_df['Discount'] < 0) & (final_df['Discount'] > max_discount), 'Pel.Cost']

    final_df['Total Fees'] = final_df['Total Fees'].astype(float).round(2)
    final_df['Discount'] = final_df['Discount'].astype(float).round(2)
    final_df = final_df.reindex(columns=config.final_columns)

    return final_df[config.final_columns]


def merge_scheduler(sched_df):
    global final_df
    sched_df.rename(columns={'Ship name': 'VESSEL_NAME'}, inplace=True)
    sched_df.rename(columns={'Tonnage': 'NET_TONNAGE'}, inplace=True)
    sched_df.rename(columns={'Estimated time to cross': 'ETC'}, inplace=True)

    sched_temp = sched_df.loc[:, [ "VESSEL_NAME", "NET_TONNAGE", "Begin", "End", "Delay","ETC"]].copy()
    sched_temp = sched_temp.drop_duplicates()

    del final_df["Begin"]
    del final_df["End"]
    del final_df["Delay"]
    del final_df["ETC"]

    final_df = pd.merge(final_df, sched_df, on =['NET_TONNAGE'], how='left')
    final_df['Begin'] = pd.to_datetime(final_df['Begin'])
    final_df['Begin'] = final_df['Begin'].dt.time
    final_df['End'] = pd.to_datetime(final_df['End'])
    final_df['End'] = final_df['End'].dt.time
    final_df['Delay'] = pd.to_datetime(final_df['Delay'])
    final_df['Delay'] = final_df['Delay'].dt.time
    final_df['ETC'] = pd.to_datetime(final_df['ETC'])
    final_df['ETC'] = final_df['ETC'].dt.time

    final_df.rename(columns={'VESSEL_NAME_x': 'VESSEL_NAME'}, inplace=True)


    return final_df[config.final_columns]


def pilotage(in_df):
    in_df['Pilotage'] = 200
    return in_df


def towage(in_df):
    mask = (in_df['NET_TONNAGE'] > 100)
    in_df.loc[in_df['NET_TONNAGE'] < 100, 'Towage Fees'] = 220
    in_df.loc[mask, 'Towage Fees'] = 220 + ((in_df.loc[mask, 'NET_TONNAGE'] - 100) * 0.25)
    return in_df


def tolls(in_df):
    maskA = (in_df['NET_TONNAGE'] > 100) & (in_df['Corinth Canal Invoice Category'] == 'Α')
    maskB = (in_df['NET_TONNAGE'] > 100) & (in_df['Corinth Canal Invoice Category'] == 'Β')
    maskC = (in_df['NET_TONNAGE'] > 100) & (in_df['Corinth Canal Invoice Category'] == 'Γ')
    maskD = (in_df['Corinth Canal Invoice Category'] == 'Δ')
    maskE = (in_df['Corinth Canal Invoice Category'] == 'Ε')
    maskSE2 = (in_df['NET_TONNAGE'] > 100) & (in_df['Corinth Canal Invoice Category'] == 'Σ2')
    maskS3 = (in_df['NET_TONNAGE'] > 100) & (in_df['Corinth Canal Invoice Category'] == 'Σ3')
    maskS4 = (in_df['NET_TONNAGE'] > 100) & (in_df['Corinth Canal Invoice Category'] == 'Σ4')
    maskS5 = (in_df['ΜΗΚΟΣ'] > 10) & (in_df['Corinth Canal Invoice Category'] == 'Σ5')
    maskST1 = (in_df['ΜΗΚΟΣ'] < 6) & (in_df['Corinth Canal Invoice Category'] == 'Σ1')
    maskST2 = (in_df['ΜΗΚΟΣ'] > 6) & (in_df['ΜΗΚΟΣ'] < 9) & (in_df['Corinth Canal Invoice Category'] == 'Σ1')
    maskST3 = (in_df['ΜΗΚΟΣ'] > 9) & (in_df['ΜΗΚΟΣ'] < 15) & (in_df['Corinth Canal Invoice Category'] == 'Σ1')
    maskST4 = (in_df['ΜΗΚΟΣ'] > 15) & (in_df['ΜΗΚΟΣ'] < 25) & (in_df['Corinth Canal Invoice Category'] == 'Σ1')
    maskST5 = (in_df['ΜΗΚΟΣ'] > 25.01) & (in_df['Corinth Canal Invoice Category'] == 'ΣΤ')

    in_df['Tolls Fees'] = 0


    in_df.loc[in_df['NET_TONNAGE'] < 100 & (in_df['Corinth Canal Invoice Category'] == 'Α'), 'Tolls Fees'] = 170
    in_df.loc[maskA, 'Tolls Fees'] = 170 + ((in_df.loc[maskA, 'NET_TONNAGE'] - 100) * 0.66)
    in_df.loc[in_df['NET_TONNAGE'] < 100 & (in_df['Corinth Canal Invoice Category'] == 'Β'), 'Tolls Fees'] = 170
    in_df.loc[maskB, 'Tolls Fees'] = 170 + ((in_df.loc[maskB, 'NET_TONNAGE'] - 100) * 0.66)
    in_df.loc[in_df['NET_TONNAGE'] < 100 & (in_df['Corinth Canal Invoice Category'] == 'Γ'), 'Tolls Fees'] = 120
    in_df.loc[maskC, 'Tolls Fees'] = 120 + ((in_df.loc[maskC, 'NET_TONNAGE'] - 100) * 0.29)
    in_df.loc[maskD, 'Tolls Fees'] = 330
    in_df.loc[maskE, 'Tolls Fees'] = 40
    in_df.loc[in_df['NET_TONNAGE'] < 100 & (in_df['Corinth Canal Invoice Category'] == 'Σ2'), 'Tolls Fees'] = 350
    in_df.loc[maskSE2, 'Tolls Fees'] = 350 + ((in_df.loc[maskSE2, 'NET_TONNAGE'] - 100) * 0.75)
    in_df.loc[maskS3, 'Tolls Fees'] = 50 + ((in_df.loc[maskS3, 'NET_TONNAGE'] - 100) * 0.35)
    in_df.loc[in_df['NET_TONNAGE'] < 100 & (in_df['Corinth Canal Invoice Category'] == 'Σ3'), 'Tolls Fees'] = 50
    in_df.loc[maskS4, 'Tolls Fees'] = 120 + ((in_df.loc[maskS4, 'NET_TONNAGE'] - 100) * 0.4)
    in_df.loc[in_df['NET_TONNAGE'] < 100 & (in_df['Corinth Canal Invoice Category'] == 'Σ4'), 'Tolls Fees'] = 120
    in_df.loc[maskS5, 'Tolls Fees'] = 18 + ((in_df.loc[maskS5, 'ΜΗΚΟΣ'] - 10) * 17)
    in_df.loc[in_df['ΜΗΚΟΣ'] < 10 & (in_df['Corinth Canal Invoice Category'] == 'Σ5'), 'Tolls Fees'] = 18
    in_df.loc[maskST1, 'Tolls Fees'] = 70
    in_df.loc[maskST2, 'Tolls Fees'] = 90
    in_df.loc[maskST3, 'Tolls Fees'] = 90 + (((in_df.loc[maskST3, 'ΜΗΚΟΣ']*1.15)-9) * 27)
    in_df.loc[maskST4, 'Tolls Fees'] = 90 + (((in_df.loc[maskST4, 'ΜΗΚΟΣ']*1.2)-9) * 30)
    in_df.loc[maskST5, 'Tolls Fees'] = 90 + (((in_df.loc[maskST5, 'ΜΗΚΟΣ']*1.25)-9) * 35)

    in_df.loc[in_df['NET_TONNAGE'].isnull(), 'Tolls Fees'] = 0

    return in_df


def total_fees(in_df):
    tr = towage(in_df)
    tr = pilotage(in_df)
    tr = tolls(in_df)
    tr['Total Transit Fees'] = tr['Pilotage'] + tr['Towage Fees'] + tr['Tolls Fees']
    tr = tr.round(decimals=2)
    return tr


app = dash.Dash(external_stylesheets=
                [dbc.themes.DARKLY])

sidebar = html.Div(
    [
        dbc.Row(
            [
                html.H6('SETTINGS',
                        style={'margin-top': '3px',
                               'text-align': 'center',
                               'backgroundColor': config.settings,
                               'color': 'white',
                               'fontWeight': 'bold', })
            ],
            style={'backgroundColor': config.settings, 'height': '27px', "width": "190px",'margin-left':'18px'},
            className=config.settings
        ),
        dbc.Row(
            [
                html.Div([
                    html.H6('Max Discount:', style={'display': 'inline-block', 'margin-top': 15, 'margin-right': 31}),
                    dcc.Input(id='maximum_discount', type='number', placeholder=max_disc,
                              style={'width': '57px', 'display': 'inline-block'}),

                    html.H6('Generations:', style={'display': 'inline-block', 'margin-top': 10, 'margin-right': 41}),
                    dcc.Input(id='generations', type='number', placeholder=config.kwargs["generations"],
                              style={'width': '57px', 'display': 'inline-block'}),

                    html.H6('Mutation:', style={'display': 'inline-block', 'margin-top': 10, 'margin-right': 62}),
                    dcc.Input(id='mutation', type='number', placeholder=config.kwargs["mutationRate"],
                              style={'width': '57px', 'display': 'inline-block'}),

                    html.H6('Elitism Rate:', style={'display': 'inline-block', 'margin-top': 10, 'margin-right': 44}),
                    dcc.Input(id='elitism', type='number', placeholder=config.kwargs["elitismRate"],
                              style={'width': '57px', 'display': 'inline-block'}),

                    html.H6('Dist.Ratio:', style={'display': 'inline-block', 'margin-top': 10, 'margin-right': 58}),
                    dcc.Input(id='distratio', type='number', placeholder=config.kwargs["dist_ratio"],
                              style={'width': '57px', 'display': 'inline-block'}),


                    html.H6('Gamma:', style={'display': 'inline-block', 'margin-top': 10, 'margin-right': 74}),
                    dcc.Input(id='gamma', type='number', placeholder=config.kwargs["gamma"],
                              style={'width': '57px', 'display': 'inline-block'}),

                    html.H6('Patience:', style={'display': 'inline-block', 'margin-top': 10, 'margin-right': 67}),
                    dcc.Input(id='patience', type='number', placeholder=config.kwargs["patience"],
                              style={'width': '57px', 'display': 'inline-block'}),


                    html.Button('Save', id='save-val', n_clicks=0,
                                style={'font-size': '17px', 'margin-top': '20px', 'backgroundColor': config.save_button,
                                       'border-radius': '8px'}),
                    html.Button('Run Scheduler', id='run_scheduler', n_clicks=0,
                                style={'font-size': '17px', 'margin-top': '20px', 'backgroundColor': config.run_button,
                                       'border-radius': '8px'}),

                    html.Hr()
                ]
                )
            ],
            style={'height': '50vh', 'width': '30vh', 'margin': '8px', 'display': 'inline-block'}),

        dbc.Row(
            [
                html.Img(id='image',src='data:image/png;base64,{}'.format(encoded_image),style={'margin-top': 30})
            ]
        ),
    ]
)

content = html.Div([
    dash_table.DataTable(
        id='live_table',
        data=get_final_df().to_dict('records'),
        columns=[{'name': i, 'id': i, 'hideable': True} for i in get_final_df().columns],
        page_action='none',
        style_as_list_view=True,
        #fixed_rows={'headers': True},
        style_cell={
            'text-align': 'center',
            'padding': '5px'
        },
        style_table={
            'padding': '0px 10px 20px 0px',
            'height': '389px',
            'overflowY': 'auto',
            'overflowX': 'scroll',
            'show-hide': 'none'
        },
        style_header={
            'backgroundColor': config.table_odd,
            'color': 'white',
            'fontWeight': 'bold',
            'height': '30px',
            'border': '2px solid white'
        },
        css=[
            {"selector": ".dash-spreadsheet-menu", "rule": "color:black; background-color:#222"},
            {"selector": ".show-hide", "rule": "color:white; background-color:#486581; fontWeight:bold;"},
            {"selector": ".show-hide:hover", "rule": "color:white; background-color:#627D98"},
            {"selector": ".show-hide-menu", "rule": "color:white; background-color:#222; fontWeight': 'bold; border:0px"},
            #{"selector": ".column-actions", "rule": ":white"}

        ],
        style_data={
            'backgroundColor': config.table_even,
            'color': 'white',
            'height': '30px',
        },
        style_data_conditional=[
            {
                'if': {
                    'row_index': 'odd',
                },
                'backgroundColor': config.table_odd,
            },
            {
                'if': {
                    'filter_query': '{Cost vs. Pel} < 0',
                    'column_id': 'Cost vs. Pel'
                },
                #'backgroundColor': config.table_discount,
                'fontWeight': 'bold',
                'color': config.table_discount
            },
            {
                'if': {
                    'filter_query': '{Cost vs. Pel} > 0',
                    'column_id': 'Cost vs. Pel'
                },
                # 'backgroundColor': config.table_discount,
                'fontWeight': 'bold',
                'color': config.table_discount_positive
            },
{
                'if': {
                    'filter_query': '{Discount} < 0',
                    'column_id': 'Discount'
                },
                # 'backgroundColor': config.table_discount,
                'fontWeight': 'bold',
                'color': config.table_discount
            },

            {
                'if': {
                    'filter_query': '{Cost vs. Pel} is blank',
                    'column_id': 'Cost vs. Pel'
                },
                'backgroundColor': config.table_null,
                'color': 'white'
            }

        ],
        style_cell_conditional=[
            {'if': {'column_id': 'IMO'},
             'maxWidth': '100px',
            'overflowX': 'visible'
            },
            {'if': {'column_id': 'VESSEL_NAME'},
             'maxWidth': '140px',
             'margin-right':'5px',
             'overflowX': 'hidden',
             'text-align': 'left'
             },
            {'if': {'column_id': 'Pilotage'},
             'maxWidth': '85px'},
            {'if': {'column_id': 'DATE'},
             'width': '7%'},
            {'if': {'column_id': 'Tolls'},
             'width': '5%'},
            {'if': {'column_id': 'Towage'},
             'width': '5%'},
            {'if': {'column_id': 'Pilotage'},
             'width': '7%'},
            {'if': {'column_id': 'Total Fees'},
             'width': '8%'},
            {'if': {'column_id': 'Cost vs. Pel'},
             },
            {'if': {'column_id': 'Begin'},
             'width': '6%'},
            {'if': {'column_id': 'End'},
             'width': '7%'},
            {'if': {'column_id': 'Delay'},
             'width': '7%'},
            {'if': {'column_id': 'ETC'},
             'width': '7%'},
            {'if': {'column_id': 'Cross No.'},
             'width': '7%'},
            {'if': {'column_id': 'Dir'},
             'width': '10%'}
        ]),
    dcc.Interval(
        id='interval-component',
        interval=18000000,
        n_intervals=0
    ),
    dcc.Graph(
        id='live-graph',
        style={
            "height": '300px',
            'width': '98%',
            'padding': '10px 10px 20px 20px'
        }
    ),
    dcc.Interval(
        id='interval2',
        interval=25200000,
        n_intervals=0
    ),
])
app.title = "Monitoring Traffic Flow within Corinth Canal, Greece"
app.layout = html.Div([
    # html.Button("Enter", style={'direction': 'right', 'padding': '10px, '}),
   # html.H5("Monitoring Traffic Flow within Corinth Canal, Greece", style={'text-align': 'center',
     #                                                                      'padding': '20px'}),
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(sidebar, width=2, className='cadetblue'),
                    dbc.Col(content, width=10)
                ],
                style={"height": "90vh"}
            ),
        ],
        fluid=True),

])


@app.callback(Output('live_table', 'data', allow_duplicate=True),
              [Input('interval-component', 'n_intervals')],
              prevent_initial_call=True
              )
def update_table(n):
    print("OK")
    config.date = pd.to_datetime(config.date) + timedelta(days=1)
    config.date = config.date.strftime("%Y-%m-%d")
    df_temp = get_final_df()
    return df_temp.to_dict('records')


@app.callback(
    Output('live_table', 'data'),
    Input('run_scheduler', 'n_clicks'),
    Input('save-val', 'n_clicks'),
    [State('maximum_discount', 'value'),
     State('generations', 'value'),
     State('mutation', 'value'),
     State('elitism', 'value'),
     State('distratio', 'value'),
     State('gamma', 'value'),
     State('patience', 'value')],
    prevent_initial_call=True
)

def update_output(n_clicks,a,value_disc,value_gen,value_mut,value_elit,value_dist,value_gamma,value_patience):
    triggered_id = ctx.triggered_id
    if triggered_id == 'save-val':
        global my_val
        global max_disc
        if (value_disc != None):
            my_val = int(value_disc)
            max_disc = my_val
            print(max_disc)
            print(my_val)
        print("VALUE_GEN: ", value_gen)
        print("VALUE_MUT: ", value_mut)
        print("VALUE_ELIT: ", value_elit)
        print("VALUE_DIST: ", value_dist)
        print("VALUE_GAMMA: ", value_gamma)
        print("VALUE_PATIENCE: ", value_patience)
        if (value_gen != None):
            config.kwargs['generations'] = value_gen
        if (value_mut != None):
            config.kwargs['mutationRate'] = value_mut
        if (value_elit != None):
            config.kwargs['elitismRate'] = value_elit
        if (value_dist != None):
            config.kwargs['dist_ratio'] = value_dist
        if (value_gamma != None):
            config.kwargs['gamma'] = value_gamma
        if (value_patience != None):
            config.kwargs['patience'] = value_patience
        print("Generations: ",config.kwargs['generations'])
        print("Mutation Rate: ",config.kwargs['mutationRate'])
        print("Elitism Rate: ",config.kwargs['elitismRate'])
        print("Dist. Ratio: ",config.kwargs['dist_ratio'])
        print("Gamma: ",config.kwargs['gamma'])
        print("Patience: ",config.kwargs['patience'])

        df_temp = get_final_df()
        return df_temp.to_dict('records')
    elif triggered_id == 'run_scheduler':
        global my_results
        proto.scheduling()
        file_path = config.folder_with_results + '/'+'ships' + str(config.date) + '.csv'
        while True:
            print('why')
            try:
                sched_df = pd.read_csv(file_path, encoding="utf-8")
                break
            except IOError:
                time.sleep(3)
        my_results=file_path
        return merge_scheduler(sched_df).to_dict('records')


@app.callback(
    Output('live-graph', 'figure'),
    Input('interval2', 'n_intervals'),
)
def MGO_scraper(num):
    global df_MGO
    temp_df = get_emea_bunker_prices()
    av = (float(temp_df['MGO']['Piraeus']) + float(temp_df['MGO']['Istanbul'])) / 2
    data = [{'TIMESTAMP': temp_df['Date']['Istanbul'], 'MGO_P': temp_df['MGO']['Piraeus'], 'MGO_I': temp_df['MGO']['Istanbul'], 'Average' : av}]
    tes = pd.DataFrame(data)
    tes['TIMESTAMP'] = tes['TIMESTAMP'].astype('datetime64[ns]')
    df_MGO = df_MGO.append(tes,ignore_index=True)
    df_MGO.to_csv('MGO.csv')
    return {
            "data": [
                {
                    "y": df_MGO['MGO_P'],
                    "x": df_MGO['TIMESTAMP'],
                    "name": "Piraeus MGO",
                    "type": "line",
                    "marker": {"color": config.graph_p},
                },
                {
                    "y": df_MGO['MGO_I'],
                    "x": df_MGO['TIMESTAMP'],
                    "name": "Instanbul MGO",
                    "type": "line",
                    "marker": {"color":config.graph_i},
                },
                {
                    "y": df_MGO['Average'],
                    "x": df_MGO['TIMESTAMP'],
                    "name": "Average",
                    "type": "line",
                    "marker": {"color": config.graph_a},
                },
            ],
            "layout": {
                "paper_bgcolor": config.graph_background,
                "plot_bgcolor": config.graph_background,
                "showlegend": True,
                "legend": dict(
                    x=0.26,
                    y=0.99,
                    traceorder='normal',
                    orientation="h",
                    font=dict(
                        color="white",
                        size=10, ),
                ),
                "xaxis": {
                    "automargin": True,
                    "title": {"text": "Date"},
                    "color": "white",
                    "gridcolor": "dimgray"
                },
                "yaxis": {
                    "automargin": True,
                    "title": {"text": "MGO"},
                    "color": "white",
                    "gridcolor": "dimgray"
                },
                "margin": {"t": 10, "l": 10, "r": 10},
            }
}

@app.callback(
    Output('image', 'src'),
    Input('live_table', 'data'),
)

def image_change(n):
    global image_path
    global encoded_image
    print("Image triggered")
    image_path = 'Vessel_Type_Donut.png'
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
    return 'data:image/png;base64,{}'.format(encoded_image)


auth = dash_auth.BasicAuth(
    app,
    {
        'maria': '1999',
        'giannis': '2000'
    }
)
if __name__ == '__main__':
    app.run_server(debug=True)