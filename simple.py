
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

    def assistant(*args, llm=None, debug=False):
        def decorator(func, *fargs, **fkwargs):
            tools = [arg for arg in args if callable(arg)]
            agent = SimpleAgent._create_function_agent(func, llm=llm, tools=tools, debug=debug)
            return agent
        return decorator

    def tool(*args, llm=None, debug=False):
        called_without_parentheses = len(args) == 1 and callable(args[0]) and llm is None and debug is False

        def decorator(func):
            if SimpleAgent._has_function_body(func):
                func.__doc__ = func.__doc__ if inspect.getdoc(func) else f"Does what the function name (s{func.__name__}) suggests."
                return func
            else:
                return SimpleAgent._create_function_agent(func, llm=llm, debug=debug)
                wrapper.__name__ = func.__name__
                wrapper.__doc__ = inspect.getdoc(func)
                return wrapper

        if called_without_parentheses:
            func = args[0]
            return decorator(func)

        elif len(args) == 0:
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

        class Proxy(Callable):
          def __init__(self, target):
            self._target = target
            self.__doc__ = f"A tool proxy for the function '{func.__name__}'." if not func.__doc__ else inspect.getdoc(func)
            self.__name__ = func.__name__
            self.__qualname__ = func.__qualname__
            self.__signature__ = inspect.signature(func)
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

from pydantic import BaseModel, Field
from typing import Any, List, Dict, Type, Union
from anytree import Node, RenderTree

class SimpleMisc:
    
    # Define ANSI escape codes for colors
    RESET = "\033[0m"
    SILVER = "\033[90m"  # Light gray or silver
    BLUE = "\033[94m"    # Bright blue
    GREEN = "\033[92m"   # Bright green
    
    def display_state(instance_or_type: Union[BaseModel, Type[BaseModel]], name: str = "Root") -> None:
        """
        Display the structure and state of a Pydantic instance or type as a tree.
    
        Args:
            instance_or_type (Union[BaseModel, Type[BaseModel]]): The Pydantic instance or type to visualize.
            name (str): Name for the root node of the tree.
    
        Returns:
            None: Prints the tree structure.
        """
        def build_tree_from_type(node: Node, model_cls: Type[BaseModel]) -> None:
            for field_name, field in model_cls.__annotations__.items():
                # Get the field type and description
                field_info = model_cls.model_fields.get(field_name)
                field_type = field_info.annotation if field_info else field
                field_description = field_info.description if field_info and hasattr(field_info, 'description') else ""
                field_name_formatted = field_name.upper() if hasattr(field_type, 'model_fields') else field_name
    
                # Create a new node for the field with color formatting
                field_display = f"{RESET}{field_name_formatted}{RESET}: {BLUE}{field_type.__name__ if hasattr(field_type, '__name__') else field_type}{RESET}"
                if field_description:
                    field_display += f'{SILVER} / {field_description}{RESET}'
    
                child_node = Node(field_display, parent=node)
    
                # If the field is another Pydantic model, recurse
                if hasattr(field_type, "model_fields"):
                    build_tree_from_type(child_node, field_type)
    
        def build_tree_from_instance(node: Node, model_instance: BaseModel) -> None:
            for field_name, field_value in model_instance.__dict__.items():
                # Get the field info and type from the model
                field_info = model_instance.__class__.model_fields.get(field_name)
                field_type = field_info.annotation if field_info else type(field_value)
                field_description = field_info.description if field_info and hasattr(field_info, 'description') else ""
                field_name_formatted = field_name.upper() if hasattr(field_type, 'model_fields') else field_name
    
                # Format the display text for the field
                field_display = (
                    f"{RESET}{field_name_formatted}{RESET}: {BLUE}{field_type.__name__ if hasattr(field_type, '__name__') else field_type}{RESET}"
                )
                if field_description:
                    field_display += f'{SILVER} / {field_description}{RESET}'
    
                # Append the value for leaf nodes
                if not hasattr(field_value, '__dict__') and not isinstance(field_value, list):
                    field_display += f" = {GREEN}{field_value}{RESET}"
    
                child_node = Node(field_display, parent=node)
    
                # Recurse for nested Pydantic models
                if isinstance(field_value, BaseModel):
                    build_tree_from_instance(child_node, field_value)
                # Handle lists of nested models or dicts
                elif isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        item_display = f"{RESET}[{i}]{RESET}"
                        if isinstance(item, BaseModel):
                            item_node = Node(item_display, parent=child_node)
                            build_tree_from_instance(item_node, item)
                        else:
                            item_display += f" = {GREEN}{item}{RESET}"
                            Node(item_display, parent=child_node)
                # Handle dictionaries
                elif isinstance(field_value, dict):
                    for key, value in field_value.items():
                        key_display = f"{RESET}{key}{RESET}: {GREEN}{value}{RESET}"
                        Node(key_display, parent=child_node)
    
        # Determine if input is a type or an instance
        root = Node(name.upper())
        if isinstance(instance_or_type, BaseModel):
            build_tree_from_instance(root, instance_or_type)
        elif isinstance(instance_or_type, type) and issubclass(instance_or_type, BaseModel):
            build_tree_from_type(root, instance_or_type)
        else:
            raise TypeError("Input must be a Pydantic BaseModel instance or type.")
    
        # Render the tree
        for pre, fill, node in RenderTree(root):
            print(f"{pre}{node.name}")
