"""
Simple test script to verify that print statements work.
"""

print("Hello, world!")
print("This is a simple test script.")
print("If you can see this, print statements are working.")

# Try to import the LightRAGManager class
try:
    from rag_system.lightrag_manager import LightRAGManager
    print("Successfully imported LightRAGManager.")
except Exception as e:
    print(f"Error importing LightRAGManager: {str(e)}")
