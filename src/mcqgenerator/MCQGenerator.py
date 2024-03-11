import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging


from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# loading the enviorment variables
load_dotenv()
# get the key
KEY = os.getenv("OPEN_AI_KEY")


# create the openai llm client
llm = ChatOpenAI(openai_api_key = KEY, model_name="gpt-3.5-turbo",
                 temperature=0.3)

# This is the template i will pass
TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choize questions for the {subject} students in \
{tone} tone. Make sure the questions are not repeated and check all the questions \
to be conforming the text as well. Make sure to format your response like RESPONSE_JSON \
below and use it as a guide.\
Ensure to make {number} of MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2 = """
You are an expert english grammarian and writer. Given a multiple choice quiz for {subject} students. \
You need to evaluate the complexity of the questions and give a acomplete analysis of the quiz. \
Only use at max 50 words for the complextiy, if the quiz is not as par with the cognitive \
and analytical abilities of the student, \
update the quiz questions which needs t be changed and change the tone such that it\
perfectly fits the student ability.
Quiz_MCQs:
{quiz}

Check for an Expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2,
)

review_chain = LLMChain(llm=llm,
                        prompt=quiz_evaluation_prompt,
                        output_key="review",
                        verbose=True)

generate_and_evaluate_quiz = SequentialChain(
                                chains=[quiz_chain, review_chain],
                                input_variables=["text", "number", "subject", "tone", "response_json"],
                                output_variables=["quiz", "review"],
                                verbose=True
                            )

