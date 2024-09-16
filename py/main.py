# from langchain_community.llms import GPT4All
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
#
# # Instantiate the model. Callbacks support token-wise streaming
# # model = GPT4All(model="/Users/andreykruglik/Library/Application Support/nomic.ai/GPT4All/nous-hermes-llama2-13b.Q4_0.gguf", n_threads=8)
#
# # Generate text
# # response = model.invoke("Once upon a time, ")
#
# # There are many CallbackHandlers supported, such as
# # from langchain.callbacks.streamlit import StreamlitCallbackHandler
#
# # callbacks = [StreamingStdOutCallbackHandler()]
# model = GPT4All(model="/Users/andreykruglik/Downloads/Llama-2-13B-Storywriter-LORA/mistral.7b.smart-lemon-cookie.gguf_v2.q4_k_m.gguf", n_threads=8)
# # prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# chain = prompt | model | StrOutputParser()
# response = chain.invoke({"topic": "bears"})
# # Generate text. Tokens are streamed through the callback manager.
# # response = model.invoke("Once upon a time, ")
# print(response)

from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Initialize models
model = GPT4All(model="/Users/andreykruglik/Downloads/Llama-2-13B-Storywriter-LORA/mistral.7b.smart-lemon-cookie.gguf_v2.q4_k_m.gguf", n_threads=8)
test_model = GPT4All(model="/Users/andreykruglik/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf", n_threads=8)

# Define prompts
template = "Write a short {language} function that will {task}. Please provide only the code without any explanation."
test_template = "Write a test for the following {language} language code:\n{code}. Output the code in a code block. Please provide only the code without any explanation or chatting."

# Create prompt templates
prompt = PromptTemplate.from_template(template)
test_prompt = PromptTemplate.from_template(test_template)

# Create the first chain
code_chain = prompt | model

# Create the second chain that uses the output of the first chain
combined_chain = RunnableSequence(
    {
        "code": code_chain,
        "language": lambda inputs: inputs['language'],
        # You can also directly pass the language if needed
    },
    test_prompt,
    test_model
)

the_code = code_chain.invoke({
   "language": args.language,
   "task": args.task,
})

# Invoke the combined chain
result = combined_chain.invoke({
    "language": args.language,
    "task": args.task,
})

print('THE FUNCTION: \n', the_code)
print('THE TEST: \n', result)

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# chain = prompt | model | StrOutputParser()
# response = chain.invoke({"topic": "bears"})
# template = ChatPromptTemplate.from_messages([
#     ("system", "You are a {language} developer with many years of experience"),
#     ("user", "Write a short {language} function that will {task}")
# ])