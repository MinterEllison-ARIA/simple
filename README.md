# Introduction
A single class intended to make it easy to develop AI agents. Based on Marvin and Langgraph.
# Features

## @prompt, classify, extract, 
A collection of convenience functions which make it easy to generate text, classify text, and extract data from text, using an LLM.

```` python
@prompt
def paraphrase(input:str) -> str:
  '''
  Rewrites the input string to have the same semantic content but a more creative and upbeat tone.
  '''
def classify_and_extract(input):
  action = classify(input, labels=['answer', 'greeting'])
  name = extract(input, str | None, "The name of the person who wrote the message")[0]
  return action, name
````

## @after, graph, display
A collection of convenience functions which make it easy to define and visualise state machine based agents using Langgraph.

```` python
class State(TypedDict):

    input: str
    action: str

class Nodes:

    @after('start')
    def classify_input(state: State):
        input = state['input']
        state['action'] = classify(input, labels=['answer', 'greeting'])
        return state

graph = SimpleGraph.graph(Nodes, State) # Generates a runnable langgraph-based agent with full streaming and debugging support
graph.display()
````

## @expect, confirm, test
A collection of convenience functions which make it easy to test any kind of code, although developed for the purpose of testing agents.

```` python
class Test

  @expect("Hello Jonathon! How can I help you", Prompts.semantic_compare)
  def test_greeting():
    response = Tests.sample_graph.invoke({"input": "Good morning. I'm Jonathon."})['response']
    confirm(response, "The response contains the name Jonathon.")
    confirm(response, "The response is a greeting.")
    return response

SimpleGraph.test(Test)
````
