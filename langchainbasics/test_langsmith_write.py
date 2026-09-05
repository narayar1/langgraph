from dotenv import load_dotenv

load_dotenv(override=True)

import os
from langsmith import Client


print("Endpoint:")
print(os.getenv("LANGSMITH_ENDPOINT"))

print("\nProject:")
print(os.getenv("LANGSMITH_PROJECT"))

print("\nWorkspace:")
print(os.getenv("LANGSMITH_WORKSPACE_ID"))

print("\nAPI key:")
key = os.getenv("LANGSMITH_API_KEY")
print(key[:10] + "..." if key else "NOT FOUND")


client = Client(
    api_key=key,
    api_url=os.getenv("LANGSMITH_ENDPOINT")
)


try:

    print("\n1. Testing READ access...")

    projects = list(client.list_projects())

    print("READ successful")

    for project in projects:
        print("   ", project.name)


    print("\n2. Testing WRITE access...")

    run_id = client.create_run(
        name="langsmith_write_test",
        run_type="chain",
        inputs={
            "message": "Hello LangSmith"
        },
        outputs={
            "response": "Write test successful"
        },
        project_name=os.getenv("LANGSMITH_PROJECT")
    )

    print("\nWRITE successful!")
    print("Run ID:", run_id)


except Exception as e:

    print("\nFAILED")
    print(type(e).__name__)
    print(e)