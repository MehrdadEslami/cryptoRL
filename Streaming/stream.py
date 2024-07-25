import requests
import time


def fetch_trades():
    print('fetch')
    url = "https://api.hitbtc.com/api/3/public/trades/BTCUSDT?sort=DESC"
    response = requests.get(url)
    return response.json()


def remove_duplicates(trades, seen):
    unique_trades = []
    for trade in trades:
        trade_id = trade['id']
        if trade_id not in seen:
            seen.add(trade_id)
            unique_trades.append(trade)
    return unique_trades


def stream_trades(buffer_size=256 * 256):
    print('stream')
    seen_trades = set()
    trade_buffer = []
    i = 0
    while i < 10:
        print('here')
        trades = fetch_trades()
        i = i + 1
        print("The %dth Iterator:" % (i))
        print(trades)
        unique_trades = remove_duplicates(trades, seen_trades)
        trade_buffer.extend(unique_trades)
        if len(trade_buffer) >= buffer_size:
            return trade_buffer[:buffer_size]
            trade_buffer = trade_buffer[buffer_size:]
        time.sleep(1000)  # Wait before fetching new trades


def stream_trade(buffer_size=256 * 256):
    print('stream')
    trade_buffer = []
    seen_trades = set()
    for i in range(10):
        print('here')
        trades = fetch_trades()
        print("The %dth Iterator:" % (i))
        print(trades)
        unique_trades = remove_duplicates(trades, seen_trades)
        trade_buffer.extend(unique_trades)

        time.sleep(2)  # Wait before fetching new trades




x = stream_trades(10*10)
print(x)
# 2415037155

#KAFKA COMMANDS
# docker exec -it kafka1 kafka-topics.sh --list --bootstrap-server localhost:29092
#
#
# docker exec -it <your_container_id> kafka-console-consumer.sh --from-beginning --bootstrap-server localhost:9092 --topic test-tp
#
#
# docker exec -it kafka1 kafka-console-producer.sh --bootstrap-server localhost:9092 --topic imagesTopic
#
#
#
# docker exec -it kafka1 kafka-topics.sh --bootstrap-server localhost:9092 --list
#
#
# create topic in kafka
# docker exec -it <your_container_id> kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 3 --topic test-tp
#
# influx
# docker exec -it influxdb influx setup -u my-user -p my-password -o my-org -b my-bucket -r 0
#
# docker exec -it influxdb influx auth list

2415698736

2415698736