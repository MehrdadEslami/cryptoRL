from kafka import KafkaProducer
import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KafkaProducer")



# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:29092',  # Use the external port
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    retries=5,  # Retry configuration
    batch_size=16384,  # Adjust batch size if needed
    linger_ms=10  # Adjust linger time if needed
)


def fetch_trades(symbol):
    url = "https://api.hitbtc.com/api/3/public/trades/" + symbol + "?limit=1000sort=DESC"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


def main():
    while True:
        try:
            print('this is before BTCUSDT')
            trades = fetch_trades(symbol='BTCUSDT')
            trades.insert(0, 'BTCUSDT')
            producer.send('trades', trades)
            print('this is after BTCUSDT')
            time.sleep(20)

            print('this is before ETHBTC')
            trades = fetch_trades(symbol='ETHBTC')
            trades.insert(0, 'ETHBTC')
            producer.send('trades', trades)
            print('this is after ETHBTC')
            time.sleep(20)

            print('this is before ETHUSDT')
            trades = fetch_trades(symbol='ETHUSDT')
            trades.insert(0, 'ETHUSDT')
            producer.send('trades', trades)
            print('this is after ETHBTC')
            producer.flush()  # Ensure all messages are sent
            logger.info("Flush(Sent) trades data to Kafka")
            time.sleep(600)

        except requests.RequestException as e:
            logger.error(f"Error fetching trades: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Producer stopped by user")
    except Exception as e:
        logger.error(f"Error in producer: {e}")
    finally:
        producer.close()
