from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, LlamaCpp
import pypyodbc
import environ

from langchain.chains import SQLDatabaseSequentialChain

env = environ.Env()
environ.Env.read_env()

API_KEY = env('OPENAI_API_KEY')

# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:{env('DBPASS')}@xxx/{env('DATABASE')}",
    include_tables=['patients'],
    sample_rows_in_table_info=5)

# setup llm"
# llm = OpenAI(temperature=0, openai_api_key=API_KEY)
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin",
    temperature=0.75,
    max_tokens=5000,
    top_p=1,
    n_ctx=10000,
    #callback_manager=callback_manager,
    verbose=True,
)



# Create db chain
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

# Setup the database chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

#db_chain = SQLDatabaseSequentialChain(llm=llm, database=db, verbose=True, top_k=3)


def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                question = QUERY.format(question=prompt)
                print(db_chain.run(question))
            except Exception as e:
                print(e)

get_prompt()
