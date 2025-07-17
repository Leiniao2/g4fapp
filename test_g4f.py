#!/usr/bin/env python3
"""
Test script to verify GPT4Free functionality
Run this before deploying to check if g4f is working
"""

import g4f
import sys
import time

def test_basic_chat():
    """Test basic chat functionality"""
    print("Testing basic chat...")
    
    try:
        messages = [
            {"role": "user", "content": "Hello, can you say hi back?"}
        ]
        
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=False
        )
        
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"Basic chat test failed: {e}")
        return False

def test_streaming():
    """Test streaming functionality"""
    print("\nTesting streaming...")
    
    try:
        messages = [
            {"role": "user", "content": "Count from 1 to 5"}
        ]
        
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if chunk:
                full_response += chunk
                print(chunk, end='', flush=True)
        
        print(f"\nFull response: {full_response}")
        return True
        
    except Exception as e:
        print(f"Streaming test failed: {e}")
        return False

def test_providers():
    """Test different providers"""
    print("\nTesting providers...")
    
    providers_to_test = ['Bing', 'You', 'Aichat']
    
    for provider_name in providers_to_test:
        try:
            if hasattr(g4f.Provider, provider_name):
                provider = getattr(g4f.Provider, provider_name)
                print(f"Testing {provider_name}...")
                
                messages = [
                    {"role": "user", "content": "Say hello from " + provider_name}
                ]
                
                response = g4f.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    provider=provider,
                    stream=False
                )
                
                print(f"{provider_name} response: {response}")
                
        except Exception as e:
            print(f"Provider {provider_name} failed: {e}")
            continue

def main():
    """Run all tests"""
    print("Starting GPT4Free tests...\n")
    
    # Test basic functionality
    if not test_basic_chat():
        print("❌ Basic chat test failed")
        return False
    else:
        print("✅ Basic chat test passed")
    
    # Test streaming
    if not test_streaming():
        print("❌ Streaming test failed")
        return False
    else:
        print("✅ Streaming test passed")
    
    # Test providers
    test_providers()
    
    print("\nAll tests completed!")
    return True

if __name__ == "__main__":
    main()
