#!/usr/bin/env python3
"""Quick test to verify Binance API connectivity."""
import requests
import json

def test_binance_api():
    """Test Binance API directly."""
    print("Testing Binance API connectivity...\n")
    
    # Test 1: Ping endpoint
    print("1. Testing Binance ping endpoint...")
    try:
        resp = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        resp.raise_for_status()
        print(f"   ✓ Ping successful: {resp.status_code}")
    except Exception as e:
        print(f"   ✗ Ping failed: {e}")
        return False
    
    # Test 2: Get server time
    print("\n2. Testing Binance server time endpoint...")
    try:
        resp = requests.get("https://api.binance.com/api/v3/time", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"   ✓ Server time: {data}")
    except Exception as e:
        print(f"   ✗ Server time failed: {e}")
        return False
    
    # Test 3: Get current SOLUSDT price
    print("\n3. Testing SOLUSDT ticker price...")
    try:
        resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"   ✓ SOLUSDT price: {data}")
    except Exception as e:
        print(f"   ✗ SOLUSDT price failed: {e}")
        return False
    
    # Test 4: Get 1d klines
    print("\n4. Testing SOLUSDT 1d klines (last 5 candles)...")
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "SOLUSDT", "interval": "1d", "limit": 5},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"   ✓ Retrieved {len(data)} klines")
        for i, candle in enumerate(data):
            print(f"     Candle {i}: Close={candle[4]}")
    except Exception as e:
        print(f"   ✗ Klines failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_binance_api()
