# Import the Pinecone library



# pc.create_index(
#     name="embeddedmodel",
#     dimension=1536, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

# index.upsert(
#     vectors=[
#         {
#             "id": "vec1", 
#             "values": [1.0, 1.5], 
#             "metadata": {"genre": "drama"}
#         }, {
#             "id": "vec2", 
#             "values": [2.0, 1.0], 
#             "metadata": {"genre": "action"}
#         }, {
#             "id": "vec3", 
#             "values": [0.1, 0.3], 
#             "metadata": {"genre": "drama"}
#         }, {
#             "id": "vec4", 
#             "values": [1.0, -2.5], 
#             "metadata": {"genre": "action"}
#         }
#     ],
#     namespace= "ns1"
# )

# response = index.query(
#     namespace="ns1",
#     vector=[0.1, 0.3],
#     top_k=2,
#     include_values=True,
#     include_metadata=True,
#     filter={"genre": {"$eq": "action"}}
# )
    
# print(response)

# embeddings = pc.inference.embed(
#     model="text-embedding-3-small",
#     inputs=[d['text'] for d in metadata],
#     parameters={"input_type": "passage", "truncate": "END"}
# )
# print(embeddings[0])

# # Wait for the index to be ready
# while not pc.describe_index(index_name).status['ready']:
#     sleep(1)

# index = pc.Index(index_name)

# vectors = []
# for d, e in zip(metadata, embeddings):
#     vectors.append({
#         "id": d['id'],
#         "values": e['values'],
#         "metadata": {'text': d['text']}
#     })

# index.upsert(
#     vectors=vectors,
#     namespace="ns1"
# )

# print(index.describe_index_stats())

# query = "Tell me about the tech company known as Apple."

# embedding = pc.inference.embed(
#     model="text-embedding-3-small",
#     inputs=[query],
#     parameters={
#         "input_type": "query"
#     }
# )

# results = index.query(
#     namespace="ns1",
#     vector=embedding[0].values,
#     top_k=3,
#     include_values=False,
#     include_metadata=True
# )

# print(results)
# embedding = get_embedding(product_description, model='text-embedding-3-small')
#    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
#    res = df.sort_values('similarities', ascending=False).head(n)

# from openai import OpenAI
# import numpy as np

# client = OpenAI()

# def normalize_l2(x):
#     x = np.array(x)
#     if x.ndim == 1:
#         norm = np.linalg.norm(x)
#         if norm == 0:
#             return x
#         return x / norm
#     else:
#         norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
#         return np.where(norm == 0, x, x / norm)


# response = client.embeddings.create(
#     model="text-embedding-3-small", input="Testing 123", encoding_format="float"
# )

# cut_dim = response.data[0].embedding[:256]
# norm_dim = normalize_l2(cut_dim)



# import pandas as pd
# import numpy as np

# df = pd.read_json('Metadata.JSON')
# df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

# from openai import OpenAI
# client = OpenAI()

# def get_embedding(text, model="text-embedding-3-small"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_json('output/embedded_1k_reviews', index=False)

from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import json
from time import sleep
# Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_5iuN6x_8jGomY4LyMdzCkdosKp81dxX6GA9cLjtMVbqeDFEZoPAWSiHBiQryejyoqknjzJ")


# Create Index
index_name = "embeddedmodel"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)


# Embed data
#data = pd.read_json("Metadata.JSON")

# json_string = '''[
#     {
#         "id": 1,
#         "user_input": "Do you have any health problems?",
#         "bot_response": "Yeah, I feel a bit tired lately."
#     },
#     {
#         "id": 2,
#         "user_input": "Are you feeling okay?",
#         "bot_response": "Well, I have a slight headache. It's been bothering me a bit."
#     },
#     {
#         "id": 3,
#         "user_input": "Is something wrong with your health?",
#         "bot_response": "Yeah, I’ve been feeling a little dizzy these past few days."
#     },
#     {
#         "id": 4,
#         "user_input": "How are you feeling today?",
#         "bot_response": "I’m a bit congested, so I think I might be catching a cold."
#     },
#     {
#         "id": 5,
#         "user_input": "Do you have any symptoms?",
#         "bot_response": "I’ve been feeling nauseous on and off today."
#     },
#     {
#         "id": 6,
#         "user_input": "Are you sick?",
#         "bot_response": "Not really, just a little runny nose and a sore throat."
#     },
#     {
#         "id": 7,
#         "user_input": "Is there something wrong with your health?",
#         "bot_response": "Yeah, my back has been hurting for a couple of days now."
#     },
#     {
#         "id": 8,
#         "user_input": "Do you feel well?",
#         "bot_response": "Not exactly, I’ve had a mild fever since this morning."
#     },
#     {
#         "id": 9,
#         "user_input": "Are you feeling unwell?",
#         "bot_response": "I’ve been feeling a little weak, like I have no energy."
#     },
#     {
#         "id": 10,
#         "user_input": "What’s bothering you?",
#         "bot_response": "I’m dealing with some muscle soreness after a workout."
#     },
#     {
#         "id": 11,
#         "user_input": "Do you feel sick?",
#         "bot_response": "I have some stomach cramps. It could be something I ate."
#     },
#     {
#         "id": 12,
#         "user_input": "How's your health today?",
#         "bot_response": "I’ve been having some shortness of breath. It’s a little concerning."
#     },
#     {
#         "id": 13,
#         "user_input": "Any health issues recently?",
#         "bot_response": "Yes, I’ve been dealing with a persistent cough for the past few days."
#     },
#     {
#         "id": 14,
#         "user_input": "Do you have any discomfort?",
#         "bot_response": "I have a bit of a sore throat, it’s been bothering me all day."
#     },
#     {
#         "id": 15,
#         "user_input": "How are you feeling right now?",
#         "bot_response": "I’m a little dizzy and lightheaded. I think I might need some rest."
#     },
#     {
#         "id": 16,
#         "user_input": "Are you experiencing any pain?",
#         "bot_response": "I’ve been having some joint pain, mostly in my knees."
#     },
#     {
#         "id": 17,
#         "user_input": "Do you feel okay today?",
#         "bot_response": "I’ve been feeling a little sluggish, like I’m coming down with something."
#     },
#     {
#         "id": 18,
#         "user_input": "Do you have any health concerns at the moment?",
#         "bot_response": "Yeah, I’ve been feeling lightheaded and weak after meals."
#     },
#     {
#         "id": 19,
#         "user_input": "Are you suffering from any symptoms?",
#         "bot_response": "I’ve been dealing with a mild headache and some body aches."
#     },
#     {
#         "id": 20,
#         "user_input": "Are you experiencing any issues with your health?",
#         "bot_response": "Yeah, I’ve been feeling a little nauseous and not hungry."
#     }
# ]'''
with open('Metadata.JSON', 'r') as file:
    data = json.load(file)
print(type(data))
import openai 
openai.api_key = "sk-svcacct-ms7hgYBtUacXP6a9iKGNMDQTydx3l6vQ_QwAlmvGLtEj5-uWmZsknU5Hj3p6C8T3BlbkFJ_EuVtVh6Bh5puZBtx0e4rWUG7uWfBaTbhI-7D3DcOsBf1EkK4QNZugFrD9xQYA" 

# dummy_text = ['text1', 'text2']
def embed(docs: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(
        input=docs,
        model="text-embedding-3-small"
    )

    embeds = [r.embedding for r in res.data]
    return embeds 

user_embeds = embed([d["user_input"] for d in data])
bot_embeds = embed([d["bot_response"] for d in data])

vectors = []
for d, u, b  in zip(data, user_embeds, bot_embeds):
    #User Vector
    vectors.append({
        "id": str(d['id']),
        "values": u,
        "metadata": {"type": 'user_input', 'original_text': d['user_input']}
    })
    #Bot Vector
    vectors.append({
        "id": str(d['id']),
        "values": b,
        "metadata": {"type": 'bot_response', 'original_text': d['bot_response']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)


### Query
query = "Tell me about the tech company known as Apple"

x = embed([query])

results = index.query(
    namespace="ns1",
    vector=x[0],
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)