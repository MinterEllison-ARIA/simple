from typing import Dict, TypedDict, Optional, Callable, Annotated
from langgraph.graph import StateGraph, START, END
import marvin
import pandas as pd

class SimpleGraph:

    classify = marvin.classify
    extract = marvin.extract
    prompt = marvin.fn

    @classmethod
    def helpers(cls):
      return cls.classify, cls.extract, cls.prompt, cls.after, cls.expect, cls.confirm

    def confirm(response: str, truth: str) -> bool:
      assert classify(response, [True, False], instructions=f"Return true if {truth}"), f'The statement "{truth}" is not true. The response evaluated was "{response}".'

    def display(self):
        from IPython.display import Image, display
        #return Image(app.get_graph(xray=True).draw_mermaid_png())
        if 'get_graph' in dir(self):
          display(Image(self.get_graph(xray=True).draw_mermaid_png()))
        else:
          display(Image(self.draw_mermaid_png()))
        return self

    def after(*edges):
      def decorator(func: Callable):
          func._edges = edges
          return staticmethod(func)
      return decorator

    def graph(node, state, compile=True):

        # Create empty graph with state type
        workflow = StateGraph(state)

        # Get all functions defined on the node class
        node_functions = [f for f in dir(node) if not f.startswith('__')]

        # First add all nodes
        for func_name in node_functions:
            func = getattr(node, func_name)
            if hasattr(func, '_edges'):
                workflow.add_node(func_name, func)

        # Then add all edges
        for func_name in node_functions:
            func = getattr(node, func_name)
            if hasattr(func, '_edges'):
              # Add edges from each source node to this node
              for source in func._edges:
                  if source == 'start' or source == START:
                      workflow.add_edge(START, func_name)
                  else:
                    workflow.add_edge(source, func_name)

        # Compile the workflow into a runnable and check for errors
        result = workflow.compile() if compile else workflow
        result.display = lambda : SimpleGraph.display(result)
        return result

    def expect(expected_value, comparator="DEFAULT"):
      def decorator(func):
          # Add a 'test' property to the function
          def test():
              actual_value = func()
              result = True if comparator == None else (actual_value == expected_value) if comparator == "DEFAULT" else comparator(actual_value, expected_value)
              return {'Test Name': func.__name__, 'Test Result': result, 'Expected': expected_value, 'Actual': actual_value}

          # Attach the 'test' function as a property of the decorated function
          func.test = test
          return func

      return decorator

    def test(cls):
      # List to store test results
      results = []

      # Iterate over all attributes of the class
      for attr_name in dir(cls):
          attr = getattr(cls, attr_name)
          # Check if the attribute is a function and has a 'test' method
          if callable(attr) and hasattr(attr, 'test'):
              # Call the 'test' method to get test results
              try:
                  test_result = attr.test()
                  results.append({
                      'Test Name': test_result['Test Name'].replace("_", " ").title(),
                      'Test Status': 'Pass' if test_result['Test Result'] else 'Fail',
                      'Expected Value': test_result['Expected'],
                      'Actual Value': test_result['Actual'],
                      'Exception': None,
                  })
              except Exception as e:
                  # Append result if test execution fails
                  results.append({
                      'Test Name': attr_name.replace("_", " ").title(),
                      'Test Status': 'Fail',
                      'Expected Value': 'N/A',
                      'Actual Value': 'N/A',
                      'Exception': str(e),
                  })

      # Create a dataframe from the results
      return pd.DataFrame(results)
