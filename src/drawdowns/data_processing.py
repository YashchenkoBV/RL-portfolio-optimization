import pandas as pd

# read data fram csv
df = pd.DataFrame(pd.read_csv('data/company-prices.csv', header=3).iloc[:, 1:], index=range(272)).rename(
    columns={'Ticker': 'Date'})

# turn prices to float
df[df.columns.difference(['Date'])] = (
    df[df.columns.difference(['Date'])].replace(',', '.', regex=True).astype('float64')
)

# calculate mean and stdev values
data = pd.DataFrame({'mean': [df[col].mean() for col in df.columns[1:]],
                     'stdev': [df[col].std() for col in df.columns[1:]]},
                    index=df.columns[1:])

if __name__ == "__main__":
    # output mean and stdev data
    print(data[:5])
