KNOTS_TO_MIN_PER_SEC = 30.8667
speed = 5.5

# Excel file with the ships and their data
data_file = '2020.xlsx'
data_file2 = 'Corinth_Canal_Transits_2020_IMO.xlsx'

# Folder where the results files will be located
folder_with_results= 'Csv_Files'

# The date for which we want results.
date='2020-01-01'

# The columns that will be written in the final file.
data_columns=["ΗΜΕΡ/ΝΙΑ ΑΦΙΞΗΣ",
           "ΟΝΟΜΑΣΙΑ ΠΛΟΙΟΥ",
           "ΑΝΑΧΩΡΗΣΗ",
           "ΠΡΟΟΡΙΣΜΟΣ",
           "ΤΥΠΟΣ ΦΟΡΤΙΟΥ",
           "ΤΥΠΟΣ ΠΛΟΙΟΥ",
           "ΕΠΩΝΥΜΙΑ ΠΕΛΑΤΗ",
           "ΕΠΩΝΥΜΙΑ ΠΡΑΚΤΟΡΑ",
           "ΣΗΜΑΙΑ",
           "ΚΑΤΕΥΘΥΝΣΗ",
           "Κ.Κ.Χ.",
           "ΜΗΚΟΣ",
           "ΒΥΘΙΣΜΑ",
           "ΩΡΑ ΑΦΙΞΗΣ",]

write_columns = ['Date', "Ship #", "Ship name","Destination Port","Deparure Port","Load Type","Vessel Type","Costumer","Agent","Flag","Length","Direction","Tonnage","Draft", "Arrival", "Begin", "End","Delay", "Estimated time to cross"]

final_columns = ["IMO", "VESSEL_NAME","DATE", "Begin","End", "Delay", "ETC", "Dir", "Cost vs Pel","Discount", "Total Fees", "Tolls", "Towage", "Pilotage","NET_TONNAGE"]

kwargs = dict(
        generations = 1,
        mutationRate = 0.85,
        elitismRate = 0.2,
        early_stopping = True,
        patience= 11,
        dist_ratio = 3,
        gamma = 0.99
        )


table_odd= '#627D98'
    #'cadetblue'
table_even='#486581'
    #'rgb(0, 102, 102)'
table_discount='darkred'
table_discount_positive='#5fcd09'
table_null='dimgray'
graph_p='#243B53'
graph_i='#486581'
graph_a='#829AB1'
graph_background='rgba(0,0,0,0)'
settings='#486581'
toggle_button='#486581'
save_button='#627D98'
run_button='#559e14'