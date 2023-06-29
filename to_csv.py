import csv
import pandas as pd
from datetime import date, datetime
import datetime
import matplotlib.pyplot as plt
import stats_pie


def convert_to_time(time):
    return datetime.timedelta(minutes=round(time / 0.05) * 0.05)


def create_csv(year, month, day, permutation, dir, cost, write_columns):
    permutation_date = date(year, month, day)

    path = dir + '/' + 'ships' + str(permutation_date) + '.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(write_columns)
        j = 1
        for i in permutation:
            name = i.name
            departure_port = i.departure_port
            destination_port = i.destination_port
            load_type = i.load_type
            vessel_type = i.vessel_type
            costumer = i.costumer
            agent = i.agent
            flag = i.flag
            length = i.length
            direction = i.direction
            tonnage = i.tonnage
            draft = i.draft
            arrival = str(permutation_date) + " " + str(convert_to_time(i.eta))
            begin = str(permutation_date) + " " + str(convert_to_time(i.departure - i.delta)) if (
                                                                                                             i.departure - i.delta) < 1440 else str(
                permutation_date + datetime.timedelta(days=1)) + " " + str(
                convert_to_time(i.departure - i.delta - 1440))
            end = str(permutation_date) + " " + str(convert_to_time(i.departure)) if (i.departure) < 1440 else str(
                permutation_date + datetime.timedelta(days=1)) + " " + str(convert_to_time(i.departure - 1440))
            delta = i.delta
            delay = (i.departure - i.delta - (i.eta))

            writer.writerow(
                [permutation_date, j, name, departure_port, destination_port, load_type, vessel_type, costumer, agent,
                 flag, length, direction, tonnage, draft, arrival, begin, end, delay, delta])
            j += 1

    df = pd.read_csv(path,encoding="ISO-8859-1")
    print(df['Ship name'])

    df[["Delay", "Estimated time to cross"]] = df[["Delay", "Estimated time to cross"]].applymap(
        lambda x: datetime.timedelta(minutes=round(x / 0.05) * 0.05))
    df["Estimated time to cross"] = df["Estimated time to cross"].apply(lambda x: str(x).split('days ')[1])
    df["Delay"] = df["Delay"].apply(lambda x: str(x).split('days ')[1])

    df.to_csv(path, index=False, header=True)

    cost = convert_to_time(cost)
    stats_pie.Donut(path,str(cost))


def records_csv(month, permutation, dir):
    path = dir + 'Records_for_month ' + month + '.csv'

    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', "Ship #", "Ship name", "Arrival", "Begin", "End", "Delay", "Estimated time to cross"])
        j = 1
        for i in permutation:
            name = i.name
            arrival = convert_to_time(i.eta)
            begin = convert_to_time((i.departure - i.delta))
            end = convert_to_time(i.departure)
            delta = convert_to_time(i.delta)
            delay = convert_to_time(i.departure - i.delta - (i.eta))

            writer.writerow([month, j, name, arrival, begin, end, delay, delta])
            j += 1

    df = pd.read_csv(path)
    # df.to_csv(path, index=False, header=True)
    print(df)


def csv_for_fitness(name, fitness):
    filename = name + '.csv'
    with open(filename, 'a') as fd:
        file = csv.writer(fd)
        file.writerow(fitness)