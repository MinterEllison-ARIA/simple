from typing import Dict, TypedDict, Optional, Callable, Annotated
from langgraph.graph import StateGraph, START, END
import marvin
import pandas as pd
from pydantic import BaseModel, ValidationError
import textwrap
import json_repair

class SimpleGraph:

    classify = marvin.classify # A helper function which allows you to classify data based on the provided labels
    prompt = marvin.fn # A decorator function which allows you to define string generator function using only docstring
    multi = type("Pipeable", (object,), {"__init__": lambda self: None, "__or__": lambda self, other: textwrap.dedent(other).strip()})() # example_str = multi | """this is a multiline string that needs indentation fixed"""

    def schema(template: BaseModel):
      """Returns a sting representation of the pydantic model schema, suitable for inclusion within prompt."""
      schema, lines = template.schema(), []
      for name, info in schema['properties'].items():
          desc, constraints = info.get('description', 'No description.'), []
          if 'enum' in info: constraints.append(f"Values: {', '.join(map(str, info['enum']))}")
          if 'exclusiveMinimum' in info: constraints.append(f"> {info['exclusiveMinimum']}")
          if 'maximum' in info: constraints.append(f"≤ {info['maximum']}")
          if 'format' in info: constraints.append(info['format'])
          line = f"{name}: {desc}" + (f" ({', '.join(constraints)})" if constraints else "")
          lines.append(line)
      return  "\n".join(lines)

    def extract(text, template=None, instructions=None, json=False, validate=True):
      """Extracts data from text. Returns tuple with the extracted data and a list of errors."""
      if not json:
        return marvin.extract(text, target=template, instructions=instructions), None
      else:
        parsed_json = json_repair.loads(text)
        if not parsed_json: return None, [(template.__name__, f"No suitable data found.")]  # No data extracted
        else:
          if not validate: return [parsed_json], None
          else:
            try: return template.parse_obj(parsed_json), None
            except ValidationError as ve:
              return parsed_json, str([(e['loc'][0], e['msg']) if e['loc'] else (template.__name__, f"No suitable data found.") for e in ve.errors()])

    @classmethod
    def helpers(cls):
      """Returns the list of helper functions available within this class"""
      return cls.classify, cls.extract, cls.prompt, cls.after, cls.expect, cls.confirm, cls.multi, cls.schema

    def confirm(response: str, truth: str) -> bool:
      """ A decorator used to define tests within a test class """
      assert classify(response, [True, False], instructions=f"Return true if {truth}"), f'The statement "{truth}" is not true. The response evaluated was "{response}".'

    def display(self):
        """Displays the graph as a mermaid diagram."""
        from IPython.display import Image, display
        if 'get_graph' in dir(self):
          display(Image(self.get_graph(xray=True).draw_mermaid_png()))
        else:
          display(Image(self.draw_mermaid_png()))
        return self

    def after(*edges):
      """ A decorator used to define a graph node within a graph definition class """
      def decorator(func: Callable):
          func._edges = edges
          return staticmethod(func)
      return decorator

    def graph(node, state, compile=True):
        """ A helper function which generates a runnable langgraph graph based on a graph definition class """

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
      """ A decorator used to define tests within a test class """
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

    def test(cls, display_only=False):
        """ A helper function which runs all of the tests within a test class """
        # List to store test results
        results = []

        # Create an empty dataframe for initial display
        df = pd.DataFrame(columns=['Test Name', 'Test Status', 'Expected Value', 'Actual Value', 'Exception'])

        # Display the dataframe initially
        if display_only==True: display_handle = display(df, display_id=True)

        # Iterate over all attributes of the class
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            # Check if the attribute is a function and has a 'test' method
            if callable(attr) and hasattr(attr, 'test'):
                try:
                    # Call the 'test' method to get test results
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

                # Update the dataframe with the current results
                df = pd.DataFrame(results)
                # clear_output(wait=True)
                if display_only: display_handle.update(df)

        # Return the final dataframe
        if not display_only: return pd.DataFrame(results)
