# verify_connection.py
from config.secrets import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET
from binance.client import Client
from binance.exceptions import BinanceAPIException

def verify():
    print("Testing Binance Futures Testnet Connection...")
    try:
        client = Client(BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET, testnet=True)
        # Fetch Futures Account Info to verify permissions and balance
        info = client.futures_account()
        
        print("\n✅ CONNECTION SUCCESSFUL!")
        print(f"Can Deposit: {info['canDeposit']}")
        print(f"Can Trade: {info['canTrade']}")
        
        print("\nBalances:")
        has_usdt = False
        for asset in info['assets']:
            if float(asset['walletBalance']) > 0:
                print(f"- {asset['asset']}: {asset['walletBalance']}")
                if asset['asset'] == 'USDT': has_usdt = True
                
        if not has_usdt:
            print("\n⚠️ WARNING: No USDT balance found. You may need to faucet some USDT for testing.")
        else:
            print("\n✅ Healthy USDT balance detected.")
            
    except BinanceAPIException as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    verify()
