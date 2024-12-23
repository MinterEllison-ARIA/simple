from simple import SimpleGraph, SimpleAgent, SimpleTest
multi, extract, tool, classify = (SimpleAgent.multi, SimpleAgent.extract, SimpleAgent.tool, SimpleAgent.classify)
graph, after = (SimpleGraph.graph, SimpleGraph.after)
test, expect, confirm = (SimpleTest.test, SimpleTest.expect, SimpleTest.confirm)

class State(TypedDict):

    input: str
    action: str
    response: str

class Tools:

    @tool
    def paraphrase(input:str) -> str:
      '''
      Rewrites the input string to have the same semantic content but a more creative and upbeat tone.
      '''

    @tool
    def semantic_compare(statement1: str, statement2: str) -> bool:
      '''
      Returns a boolean indicating whether the two input statements are similar in terms of when they might be uttered in a conversation,
      and whether they would have a similar impact on that conversation.
      Return false if the two statements would have a materially different impact on a conversation, given the same contex.
      '''

class Nodes:

    @after('start')
    def classify_input(state: State):
        input = state['input']
        state['action'] = classify(input, labels=['answer', 'greeting'])
        return state

    @after('classify_input')
    def respond_greeting(state: State):
        if state['action'] == 'greeting':
          input = state['input']
          name = extract(input, str | None, "The name of the person who wrote the message")[0]
          state["response"] = Prompts.paraphrase("Hello! How can I help you today?" if name == None else f"Hello {name}! How can I help you today?")
          return state

    @after('classify_input')
    def respond_answer(state: State):
        if state['action'] == 'answer':
          state["response"] = "The answer is always 42"
          return state

    @after('respond_greeting', 'respond_answer')
    def send_email(state: State):
      pass

class Tests:

    sample_graph = SimpleGraph.graph(Nodes, State)

    @expect('The answer is always 42')
    def test_answer():
        return Tests.sample_graph.invoke({"input": "What is 2 + 2?"})['response']

    @expect("Hello Jonathon! How can I help you", Prompts.semantic_compare)
    def test_greeting():
      response = Tests.sample_graph.invoke({"input": "Good morning. I'm Jonathon."})['response']
      confirm(response, "The response contains the name Jonathon")
      confirm(response, "The response is a greeting")
      return response

SimpleGraph.test(Tests, display_only=True)
Tests.sample_graph.display()
