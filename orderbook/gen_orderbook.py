import heapq
class Order:
    def __init__(self, order_id, is_buy, quantity, price):
        self.id = order_id  # 订单ID
        self.is_buy = is_buy # 是否为买入订单
        self.quantity = quantity # 数量
        self.price = price
class PriceLevel:
    def __init__(self, price):
        self.price = price # 价格
        self.orders = []# 该价格水平的订单列表
        self.total_quantity = 0 # 该价格水平的总数量
    def add_order(self, order):
        self.orders.append(order) # 将订单添加到订单列表
        self.total_quantity += order.quantity # 更新总数量
    def remove_order(self, order_id):
        for idx, order in enumerate(self.orders):
            if order.id == order_id:
                self.total_quantity -= order.quantity # 更新总数量
                del self.orders[idx] # 从订单列表中删除订单
                break
class OrderBook:
    def __init__(self):
        self.bids = [] # 买入订单的堆（最大堆，使用负价格）
        self.asks = [] # 卖出订单的堆（最小堆）
        self.bid_price_map = {} # 买入价格到PriceLevel的映射
        self.ask_price_map = {} # 卖出价格到PriceLevel的映射
        self.order_id_map = {}  # 订单ID到订单的映射
        self.current_order_id = 0 # 当前订单ID
    def add_order(self, is_buy, quantity, price):
        self.current_order_id += 1
        order = Order(self.current_order_id, is_buy, quantity, price)
        self.order_id_map[order.id] = order # 将订单添加到订单ID映射
        if is_buy:
           if price in self.bid_price_map:
               price_level = self.bid_price_map[price]
           else:
                price_level = PriceLevel(price)
                heapq.heappush(self.bids, -price_level.price)# 使用负价格插入最大堆
                self.bid_price_map[price] = price_level
                price_level.add_order(order)
        else:
            if price in self.ask_price_map:
                price_level = self.ask_price_map[price]
            else:
                price_level = PriceLevel(price)
                heapq.heappush(self.asks, price_level.price) # 插入最小堆
                self.ask_price_map[price] = price_level
                price_level.add_order(order)
    def cancel_order(self, order_id):
        if order_id in self.order_id_map:
            order = self.order_id_map[order_id]
            if order.is_buy:
                price_level = self.bid_price_map[order.price]
                price_level.remove_order(order_id)
                if price_level.total_quantity == 0:
                    del self.bid_price_map[order.price]
            else:
                price_level = self.ask_price_map[order.price]
                price_level.remove_order(order_id)
                if price_level.total_quantity == 0:
                    del self.ask_price_map[order.price]
            del self.order_id_map[order_id] # 从订单ID映射中删除订单
    def execute_trades(self):
        while True:
            if self.bids and self.asks:
                best_bid_price = -self.bids[0]# 获取最高买入价
                best_ask_price = self.asks[0] # 获取最低卖出价
                if best_bid_price not in self.bid_price_map or best_ask_price not in self.ask_price_map:
                    continue # 如果价格水平不再活跃，则跳过
                if best_bid_price >= best_ask_price:
                    bid_level = self.bid_price_map[best_bid_price]
                    ask_level = self.ask_price_map[best_ask_price]
                    if not bid_level.orders or not ask_level.orders:
                        continue# 如果该价格水平没有订单，则跳过
                    bid_order = bid_level.orders[0]
                    ask_order = ask_level.orders[0] # 确定成交数量
                    trade_qty = min(bid_order.quantity, ask_order.quantity)# 执行交易
                    print(f"Trade executed: {trade_qty} at price {best_ask_price}")# 更新数量
                    bid_order.quantity -= trade_qty
                    ask_order.quantity -= trade_qty# 移除已成交的订单
                    if bid_order.quantity == 0:
                        bid_level.remove_order(bid_order.id)
                    if bid_level.total_quantity == 0:
                        del self.bid_price_map[best_bid_price]
                    if ask_order.quantity == 0:
                        ask_level.remove_order(ask_order.id)
                    if ask_level.total_quantity == 0:
                        del self.ask_price_map[best_ask_price]
                    else:
                        break
                else:
                    break
    def print_order_book(self):
        print("Bids:")
        for price in sorted(self.bid_price_map.keys(), reverse=True):
            level = self.bid_price_map[price]
            print(f"Price: {price}, Quantity: {level.total_quantity}, Orders: {level.orders}")
            print("\nAsks:")
            for price in sorted(self.ask_price_map.keys()):
                level = self.ask_price_map[price]
                print(f"Price: {price}, Quantity: {level.total_quantity}, Orders: {level.orders}")

# 主函数测试订单簿
if __name__ == "__main__":
    ob = OrderBook() # 添加买入订单
ob.add_order(is_buy=True, quantity=100, price=10)
ob.add_order(is_buy=True, quantity=200, price=10)
ob.add_order(is_buy=True, quantity=150, price=11)    # 添加卖出订单    ob.add_order(is_buy=False, quantity=150, price=11)    ob.add_order(is_buy=False, quantity=100, price=10)    # 打印订单簿    ob.print_order_book()    # 执行交易    ob.execute_trades()    # 交易后打印订单簿    ob.print_order_book()    # 取消订单    ob.cancel_order(2)    # 取消订单后打印订单簿    ob.print_order_book()