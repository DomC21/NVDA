from config import (
    UNUSUAL_WHALES_API_KEY,
    POLYGON_API_KEY,
    ALPHA_VANTAGE_API_KEY,
    SYMBOL
)
import requests
from datetime import datetime, timedelta


def test_unusual_whales_api():
    try:
        # Try the alerts endpoint which is commonly available
        url = "https://unusualwhales.com/api/alerts"
        headers = {
            "Accept": "application/json",
            "Authorization": UNUSUAL_WHALES_API_KEY
        }
        params = {
            "symbol": SYMBOL,
            "limit": 1
        }
        
        print(f"\nTesting Unusual Whales API connection...")
        print(f"API Key: {UNUSUAL_WHALES_API_KEY}")
        
        response = requests.get(url, headers=headers, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code != 200:
            print(f"Response Text: {response.text}")
            print(f"Request URL: {url}")
            print(f"Request Headers: {headers}")
            print(f"Request Params: {params}")
            
            # Try alternative endpoint
            alt_url = "https://unusualwhales.com/api/v2/options/flow"
            alt_response = requests.get(alt_url, headers=headers, params=params)
            print(f"\nAlternative endpoint status: {alt_response.status_code}")
            if alt_response.status_code != 200:
                print(f"Alt Response Text: {alt_response.text}")
            else:
                return True
                
        return response.status_code == 200
    except Exception as e:
        print(f"Unusual Whales API Exception: {str(e)}")
        return False


def test_polygon_api():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/1/day/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        )
        params = {"apiKey": POLYGON_API_KEY}
        response = requests.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        print(f"Polygon API Exception: {str(e)}")
        return False


def test_alpha_vantage_api():
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": SYMBOL,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        print(f"Alpha Vantage API Exception: {str(e)}")
        return False


def main():
    print("Testing API integrations...")
    
    apis = {
        "Unusual Whales": test_unusual_whales_api,
        "Polygon": test_polygon_api,
        "Alpha Vantage": test_alpha_vantage_api
    }
    
    results = {}
    for name, test_func in apis.items():
        try:
            success = test_func()
            results[name] = "Success" if success else "Failed"
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    
    print("\nResults:")
    for api, status in results.items():
        print(f"{api}: {status}")


if __name__ == "__main__":
    main()
