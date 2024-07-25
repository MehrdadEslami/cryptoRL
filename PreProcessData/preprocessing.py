from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def query_trades(symbol, start_time, end_time, bucket, org, token, url="http://localhost:8086"):
    client = InfluxDBClient(url=url, token=token, org=org)
    query = f'''
        from(bucket: "{bucket}")
          |> range(start: {start_time}, stop: {end_time})
          |> filter(fn: (r) => r["_measurement"] == "trades")
          |> filter(fn: (r) => r["symbol"] == "{symbol}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "price", "quantity", "side"])
    
        '''
    tables = client.query_api().query(query)
    client.close()

    records = []
    print('in query_trade bu query:')
    print(query)
    print('the result of query:')
    print(tables)
    for table in tables:
        for record in table.records:
            print('in table query', record['_time'])
            records.append((record["_time"], record["price"], record["quantity"],
                            record["side"] ))

    df = pd.DataFrame(records, columns=['time', 'price', 'quantity', 'side'])
    return df

# def query_count(symbol, start_time, end_time, bucket, org, token, url="http://localhost:8086"):
    

def trades_to_image(trades, buffer_size, symbol):
    # Initialize image channels
    buy_channel = np.zeros((buffer_size, buffer_size))
    sell_channel = np.zeros((buffer_size, buffer_size))
    price_channel = np.zeros((buffer_size, buffer_size))

    for i, trade in trades.iterrows():
        if i >= buffer_size:
            break
        price = trade['price']
        quantity = trade['quantity']
        side = trade['side']

        row = i // buffer_size
        col = i % buffer_size

        price_channel[row, col] = price
        if side == 'buy':
            buy_channel[row, col] = quantity
        else:
            sell_channel[row, col] = quantity

    image = np.stack((buy_channel, sell_channel, price_channel), axis=2)

    # plt.imshow(image)
    # plt.title(f"Trade Data for {symbol}")
    # plt.xlabel("Trades")
    # plt.ylabel("Buffer")
    # plt.show()

    return image
