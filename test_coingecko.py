#!/usr/bin/env python3
"""Test CoinGecko API."""
import requests

print("Testing CoinGecko API...\n")

# Test 1: Simple price
print("1. Testing simple price endpoint...")
try:
    resp = requests.get(
        'https://api.coingecko.com/api/v3/simple/price',
        params={'ids': 'solana', 'vs_currencies': 'usd'},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"   ✓ Current price: ${data['solana']['usd']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Historical market chart
print("\n2. Testing market_chart endpoint (30 days)...")
try:
    resp = requests.get(
        'https://api.coingecko.com/api/v3/coins/solana/market_chart',
        params={'vs_currency': 'usd', 'days': 30, 'interval': 'daily'},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    prices = data.get('prices', [])
    print(f"   ✓ Got {len(prices)} daily price points")
    if prices:
        print(f"     Latest: {prices[-1]}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Large history
print("\n3. Testing market_chart endpoint (1825 days max)...")
try:
    resp = requests.get(
        'https://api.coingecko.com/api/v3/coins/solana/market_chart',
        params={'vs_currency': 'usd', 'days': 1825, 'interval': 'daily'},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    prices = data.get('prices', [])
    print(f"   ✓ Got {len(prices)} daily price points")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n✓ CoinGecko API tests complete!")
