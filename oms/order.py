class Order:
    def __init__(self, step: int, side: str, trade_type: str, trading_pair: str, quantity: float, price: float):
        self.step = step
        self.side = side
        self.trade_type = trade_type
        self.trading_pair = trading_pair
        self.quantity = quantity
        self.price = price

    def execute(self):
        print(f'Executing {self.side} order for {self.quantity} of {self.trading_pair} at {self.price}')
