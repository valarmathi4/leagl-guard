#!/usr/bin/env python3
"""
Test script to verify Gemini and WatsonX AI provider integration
"""
import os
import sys

# Test AI provider initialization
print("=" * 60)
print("Testing AI Provider Integration")
print("=" * 60)

try:
    from service.ContractAnalyzerService import ContractAnalyzerService
    
    # Initialize the service
    print("\n1. Initializing ContractAnalyzerService...")
    service = ContractAnalyzerService()
    
    # Check which AI provider is active
    print(f"\n2. Active AI Provider: {service.ai_provider or 'None (Fallback mode)'}")
    
    # Check Gemini client
    if service.gemini_client:
        print("   ✅ Gemini Client: Available")
        print(f"      Model: {service.gemini_client.config.model_name}")
    else:
        print("   ❌ Gemini Client: Not configured")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("      Reason: GEMINI_API_KEY not set in environment")
        else:
            print("      Reason: Configuration or initialization error")
    
    # Check WatsonX client
    if service.watsonx_client:
        print("   ✅ WatsonX Client: Available")
        print(f"      Model: {service.watsonx_client.config.model_id}")
    else:
        print("   ❌ WatsonX Client: Not configured")
        ibm_key = os.getenv("IBM_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        if not ibm_key or not project_id:
            print("      Reason: IBM_API_KEY or WATSONX_PROJECT_ID not set")
        else:
            print("      Reason: Configuration or initialization error")
    
    print("\n3. AI Provider Priority:")
    print("   1st Priority: Google Gemini")
    print("   2nd Priority: IBM WatsonX Granite")
    print("   Fallback: Intelligent rule-based analysis")
    
    print("\n" + "=" * 60)
    if service.ai_provider:
        print(f"✅ SUCCESS: {service.ai_provider.upper()} is ready for contract analysis!")
    else:
        print("⚠️  INFO: No AI provider configured, will use fallback analysis")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
