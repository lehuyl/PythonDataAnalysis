import numpy as np
import pandas as pd


def get_precip_data():
    return pd.read_csv('precipitation.csv', parse_dates=[2])


def date_to_month(d):
    # You may need to modify this function, depending on your data types.
    
    
    #4 integers and 2 integers?
    return '%04i-%02i' % (d.year, d.month) # try out apply function


def getName(data):
    return data.name

def pivot_months_pandas(data):
    """
    Create monthly precipitation totals for each station in the data set.
    
    This should use Pandas methods to manipulate the data.
    """
    # ...

    month = data.date.apply(date_to_month)
    dfm = pd.DataFrame({'month': month})
    data['month'] = dfm['month']
    
    # data for precipitation totals
    monthGroups = data.groupby(['name', 'month'])
    total = monthGroups.aggregate('sum').reset_index()
    
    #data for counts
    #need to count occurrences per month
    
    monthly = total.pivot(index='name', columns='month', values='precipitation')
    
    
    countGroups = data.groupby(['name', 'month'])
    count = countGroups.aggregate('count').reset_index()
    counts = count.pivot(index='name', columns='month')
    
    
    return monthly, counts


def pivot_months_loops(data):
    """
    Create monthly precipitation totals for each station in the data set.
    
    This does it the hard way: using Pandas as a dumb data store, and iterating in Python.
    """
    # Find all stations and months in the data set.
    stations = set()
    months = set()
    for i,r in data.iterrows():
        stations.add(r['name'])
        m = date_to_month(r['date'])
        months.add(m)

    # Aggregate into dictionaries so we can look up later.
    stations = sorted(list(stations))
    row_to_station = dict(enumerate(stations))
    station_to_row = {s: i for i,s in row_to_station.items()}
    
    months = sorted(list(months))
    col_to_month = dict(enumerate(months))
    month_to_col = {m: i for i,m in col_to_month.items()}

    # Create arrays for the data, and fill them.
    precip_total = np.zeros((len(row_to_station), 12), dtype=np.uint)
    obs_count = np.zeros((len(row_to_station), 12), dtype=np.uint)

    for _, row in data.iterrows():
        m = date_to_month(row['date'])
        r = station_to_row[row['name']]
        c = month_to_col[m]

        precip_total[r, c] += row['precipitation']
        obs_count[r, c] += 1

    # Build the DataFrames we needed all along (tidying up the index names while we're at it).
    totals = pd.DataFrame(
        data=precip_total,
        index=stations,
        columns=months,
    )
    totals.index.name = 'name'
    totals.columns.name = 'month'
    
    counts = pd.DataFrame(
        data=obs_count,
        index=stations,
        columns=months,
    )
    counts.index.name = 'name'
    counts.columns.name = 'month'
    
    return totals, counts


def main():
    data = get_precip_data()
    totals, counts = pivot_months_loops(data)
    totals.to_csv('totals.csv')
    counts.to_csv('counts.csv')
    np.savez('monthdata.npz', totals=totals.values, counts=counts.values)


if __name__ == '__main__':
    main()
