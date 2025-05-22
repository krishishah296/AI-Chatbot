# from pinecone import Pinecone, ServerlessSpec

# pc = Pinecone(api_key="")

# index_name = "quickstart"

# pc.create_index(
#     name=index_name,
#     dimension=2, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )
import openai
from openai import OpenAIError
# from openai import OpenAI, OpenAIError
# client = OpenAI(api_key="sk-svcacct-ms7hgYBtUacXP6a9iKGNMDQTydx3l6vQ_QwAlmvGLtEj5-uWmZsknU5Hj3p6C8T3BlbkFJ_EuVtVh6Bh5puZBtx0e4rWUG7uWfBaTbhI-7D3DcOsBf1EkK4QNZugFrD9xQYA");
openai.api_key="sk-svcacct-ms7hgYBtUacXP6a9iKGNMDQTydx3l6vQ_QwAlmvGLtEj5-uWmZsknU5Hj3p6C8T3BlbkFJ_EuVtVh6Bh5puZBtx0e4rWUG7uWfBaTbhI-7D3DcOsBf1EkK4QNZugFrD9xQYA"
# completion = client.chat.completions.create(
#   model="o1-mini-2024-09-12",
#   messages=[
#     {"role": "user", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ],
#     max_completion_tokens = 200,
#     stream=False,
#     store=True,
# )

def chat_with_openai():
    print("Welcome to the chat! Type 'quit' to end the conversation.\n")
    messages = []
    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to quit
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        messages.append(
        {"role": "user", "content": user_input}
        )
        try:
            # response = client.chat.completions.create(
            # model="o1-mini-2024-09-12",
            # messages=[
            #   {"role": "user", "content": user_input}
            #  ],

            #   max_completion_tokens = 200,
            #   stream=False,
            #   store=True,
            #   n=1,                # Return one response
            #   stop=None,          # Don't use stop tokens
            #   temperature=0.7,
            # )

            response = openai.ChatCompletion.create(
                model="o1-mini-2024-09-12",  # Ensure the model is valid for your account
                messages=messages,  # Pass the entire conversation history
               # max_tokens=200,     # Limit the number of tokens in the response
                n=1,                # Return one response
                #temperature=0.7,    # Control the randomness of the response
            )

            ai_response = response.choices[0].message.content
            print("AI: {}".format(ai_response))
        # Get the generated text from the response
            messages.append({"role": "assistant", "content": ai_response})
        except OpenAIError as e:
            print("Error occurred: {}".format(e))
            continue

# Run the chat function
chat_with_openai()