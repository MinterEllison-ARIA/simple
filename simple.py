
from langgraph.graph import StateGraph, START, END

class SimpleGraph:

    def _display(self):
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
          str_edges = [edge.__name__ if isinstance(edge, Callable) else ('start' if edge == None else edge) for edge in edges] if edges else ['start']
          func._edges = str_edges
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
        result.display = lambda : SimpleGraph._display(result)
        return result

import marvin
import pandas as pd

class SimpleTest:

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

    def confirm(response: str, truth: str) -> bool:
      """ A decorator used to define tests within a test class """
      assert marvin.classify(response, [True, False], instructions=f"Return true if {truth}"), f'The statement "{truth}" is not true. The response evaluated was "{response}".'
      return True

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

import inspect
import textwrap
import re
import marvin
import json_repair

from typing import Any, Dict, Literal, Optional, Callable, TypedDict, Annotated
from pydantic import BaseModel, ValidationError
from enum import Enum
from functools import wraps, partial
from jinja2 import Template
from marvin.utilities.pydantic import cast_type_or_alias_to_model
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from enum import Enum

class SimpleAgent:

    multi = type("Pipeable", (object,), {"__init__": lambda self: None, "__or__": lambda self, other: textwrap.dedent(other).strip()})() # example_str = multi | """this is a multiline string that needs indentation fixed"""
    classify = marvin.classify # A helper function which allows you to classify data based on the provided labels

    def extract(text=None, template=None, instructions=None, assume_json_input=False, validate_template=True):
      """Extracts data from text. Returns tuple with the extracted data plus a list of validation errors if validation fails."""
      if not assume_json_input:
        return marvin.extract(text, target=template, instructions=instructions), None
      else:
        parsed_json = json_repair.loads(text)
        if not parsed_json: return None, [(template.__name__, f"No suitable data found.")]  # No data extracted
        else:
          if not validate_template: return [parsed_json], None
          else:
            try: return template.parse_obj(parsed_json), None
            except ValidationError as ve:
              return parsed_json, str([(e['loc'][0], e['msg']) if e['loc'] else (template.__name__, f"No suitable data found.") for e in ve.errors()])

    def tool(*args, llm = None, subtools = [], **kwargs) -> Callable:
        """ 
        Decorator which generates agents from docstrings. 
        Accepts functions as inputs which become subtools. 
        Can also be used to mark a normal python function as suitable for use as a subtool 
        """
        
        has_function_body = SimpleAgent._has_function_body
        first_arg = args[0] if args else None

        if len(args) == 1 and not hasattr(first_arg, '__is_tool__'):
            if has_function_body(first_arg):
                first_arg.__is_tool__ = True
                return first_arg
            return SimpleAgent._create_function_agent(first_arg, llm=llm)

        def decorator(func):
            if has_function_body(func):
                func.__is_tool__ = True
                return func
            tools = [arg for arg in args if callable(arg)]
            for tool in tools:
                if not inspect.getdoc(tool):
                    tool.__doc__ = f"Does what the function name ({tool.__name__}) suggests."
            return SimpleAgent._create_function_agent(func, llm=llm, tools=tools + subtools)
        return decorator

    def _has_function_body(func) -> bool: 
      func_source, func_doc = inspect.getsource(func), func.__doc__ if func.__doc__ else ""
      after_doc_index = func_source.find(func_doc) + len(func_doc)
      return bool(re.search(r'[^\s\'"]', func_source[after_doc_index:]))

    @staticmethod
    def _create_function_agent(func: Callable, llm=None, tools=None, debug=False):

        # set defaults
        llm = llm if llm else ChatOpenAI(model="gpt-4o-mini")
        tools = tools if tools else []

        # generate a pydantic class representing the return signature
        return_type = inspect.signature(func).return_annotation
        return_type_imputed = \
            str if isinstance(return_type, (str, type(None))) or return_type is inspect.Signature.empty \
            else Enum("Labels", {"v" + str(i): label for i, label in enumerate(return_type)}) if isinstance(return_type, list) \
            else return_type
        return_type_pydantic = cast_type_or_alias_to_model(return_type_imputed)
        parser = PydanticOutputParser(pydantic_object=return_type_pydantic)
        format_instructions = parser.get_format_instructions()
        function_definition = f'def {func.__name__}{str(inspect.signature(func))})\n"""\n{inspect.getdoc(func)}\n"""'
        agent_schema = type("agent_schema", (AgentState,), {"__annotations__": {"input": dict, "output": return_type_pydantic}})

        def append_function_inputs(state: AgentState):
            function_inputs = state['input']
            function_definition_renderered = Template(function_definition).render(function_inputs=function_inputs) if function_inputs else function_definition
            system_message_template = ChatPromptTemplate.from_messages(
                [("system", Template(SimpleAgent.tool.SYSTEM_PROMPT_TEMPLATE).render(function_definition=function_definition_renderered)),
                ("placeholder", "{messages}")])
            system_messages = system_message_template.invoke({"messages": state["messages"]}).to_messages()
            user_message = [("user", Template(SimpleAgent.tool.MESSAGE_PROMPT_TEMPLATE).render(function_inputs=state['input'], format_instructions=format_instructions))]
            return system_messages + user_message

        class Proxy:
          def __init__(self, target): 
            self._target = target
            self.__doc__ = f"A tool proxy for the function '{func.__name__}'.\n\nFurther details:\n{func.__doc__ or 'None'}"
          def __getattr__(self, name): return getattr(self._target, name)
          def __call__(self, *args: Any, **kwds: Any) -> Any:
            mapped_inputs = dict(inspect.signature(func).bind(*args,**kwds).apply_defaults() or inspect.signature(func).bind(*args,**kwds).arguments)
            response_messages = self._target.invoke({'input': mapped_inputs if mapped_inputs else  None})
            str_result = response_messages["messages"][-1].content
            return parser.parse(str_result).output

        tool_agent = create_react_agent(llm, tools, state_schema=agent_schema, state_modifier=append_function_inputs, debug=debug)
        return Proxy(tool_agent)

    # Define the Jinja2 templates for @tool
    tool.SYSTEM_PROMPT_TEMPLATE = multi | """
        Your job is to generate likely outputs for a Python function with the
        following definition:

        {{ function_definition }}

        The user will provide function inputs (if any) and you must respond with
        the most likely result.

        e.g. `list_fruits(n: int) -> list[str]` (3) -> "apple", "banana", "cherry"
    """

    tool.MESSAGE_PROMPT_TEMPLATE = multi | """
          ## Function inputs

          {% if function_inputs -%}
          The function was called with the following inputs:
          {% for arg, value in function_inputs.items() %}
          - {{ arg }}: {{ value }}
          {% endfor %}
          {% else %}
          The function was not called with any inputs.
          {% endif %}

          {% if format_instructions -%}
          ## Additional Context

          I also preprocessed some of the data and have this additional context for you to consider:

          {{ format_instructions }}
          {% endif %}

          What is the function's output?

          |ASSISTANT|

          The output is
      """

SimpleAgent.assistant = SimpleAgent.tool
