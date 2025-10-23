"""
Example demonstrating async/await support in the REPL environment.

This example shows how the REPLEnv now safely supports both synchronous
and asynchronous code execution.
"""

import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Set your actual API key

from rlm.repl import REPLEnv


def main():
    # Create a REPL environment
    repl = REPLEnv()

    print("=" * 60)
    print("Example 1: Synchronous Code (still works!)")
    print("=" * 60)

    result = repl.code_execution("""
x = 10
y = 20
x + y
""")
    print(f"Output: {result.stdout}")
    print(f"Variables: x={result.locals['x']}, y={result.locals['y']}")
    print()

    print("=" * 60)
    print("Example 2: Async Code with await")
    print("=" * 60)

    result = repl.code_execution("""
import asyncio

async def fetch_data():
    await asyncio.sleep(0.1)
    return "Hello from async!"

# Top-level await now works!
data = await fetch_data()
print(f"Fetched: {data}")
""")
    print(f"Output: {result.stdout}")
    print(f"Data variable: {result.locals['data']}")
    print()

    print("=" * 60)
    print("Example 3: Async Expression (auto-printed)")
    print("=" * 60)

    result = repl.code_execution("""
import asyncio

async def compute():
    await asyncio.sleep(0.1)
    return 42

# This expression will be evaluated and printed
await compute()
""")
    print(f"Output: {result.stdout}")
    print()

    print("=" * 60)
    print("Example 4: Mixed sync and async")
    print("=" * 60)

    # First, sync code
    result1 = repl.code_execution("""
base_value = 100
""")

    # Then, async code using the sync variable
    result2 = repl.code_execution("""
import asyncio

async def multiply(x):
    await asyncio.sleep(0.1)
    return x * 2

result = await multiply(base_value)
print(f"Result: {result}")
""")
    print(f"Output: {result2.stdout}")
    print(f"Result: {result2.locals['result']}")
    print()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
