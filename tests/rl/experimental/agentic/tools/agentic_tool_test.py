import asyncio
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.experimental.agentic.tools import base_tool
from tunix.rl.experimental.agentic.tools import calculator_tool
from tunix.rl.experimental.agentic.tools import tool_manager


class BaseToolTest(absltest.TestCase):

  def test_tool_call_creation(self):
    tool_call = base_tool.ToolCall(
        name="test_tool", arguments={"arg1": "value1"}
    )
    self.assertEqual(tool_call.name, "test_tool")
    self.assertEqual(tool_call.arguments, {"arg1": "value1"})

  def test_tool_output_repr(self):
    # Test with error
    error_output = base_tool.ToolOutput(
        name="test", error="Something went wrong"
    )
    self.assertEqual(str(error_output), "Error: Something went wrong")

    # Test with dict output
    dict_output = base_tool.ToolOutput(name="test", output={"key": "value"})
    self.assertEqual(str(dict_output), '{"key": "value"}')

    # Test with list output
    list_output = base_tool.ToolOutput(name="test", output=[1, 2, 3])
    self.assertEqual(str(list_output), "[1, 2, 3]")

    # Test with string output
    str_output = base_tool.ToolOutput(name="test", output="Success")
    self.assertEqual(str(str_output), "Success")

    # Test with None output
    none_output = base_tool.ToolOutput(name="test", output=None)
    self.assertEqual(str(none_output), "")

  def test_base_tool_abstract_methods(self):
    # We test that a subclass without implementing abstract methods fails.
    with self.assertRaises(TypeError):

      # pylint: disable=abstract-class-instantiated
      class IncompleteTool(base_tool.BaseTool):
        pass

      IncompleteTool(name="incomplete", description="...")

  def test_base_tool_apply_not_implemented(self):

    class TestTool(base_tool.BaseTool):

      @property
      def get_json_schema(self):
        return {}

    tool = TestTool(name="test", description="A test tool.")
    with self.assertRaises(NotImplementedError):
      tool.apply()

  def test_base_tool_apply_async_not_implemented(self):

    class TestTool(base_tool.BaseTool):

      @property
      def get_json_schema(self):
        return {}

      def apply(self, **kwargs):
        # This is implemented, but apply_async should not delegate to it.
        return base_tool.ToolOutput(name=self.name, output="sync result")

    tool = TestTool(name="test", description="A test tool.")
    with self.assertRaises(NotImplementedError):
      asyncio.run(tool.apply_async())


class ToolManagerTest(parameterized.TestCase):

  # Test docstring fallback
  class NoDocstringTool(base_tool.BaseTool):

    @property
    def get_json_schema(self):
      return {}

  def setUp(self):
    super().setUp()
    self.tool_map = {"calculator": calculator_tool.CalculatorTool}
    self.manager = tool_manager.ToolManager(self.tool_map)

  def test_initialization(self):
    self.assertIn("calculator", self.manager.names)
    self.assertIsInstance(
        self.manager._tool_dict["calculator"], calculator_tool.CalculatorTool
    )
    # Test docstring fallback
    manager = tool_manager.ToolManager(
        {"no_doc": self.NoDocstringTool}, desc_fallback="fallback"
    )
    self.assertEqual(manager._tool_dict["no_doc"].description, "fallback")

  def test_get_json_schema(self):
    schemas = self.manager.get_json_schema
    self.assertLen(schemas, 1)
    self.assertEqual(schemas[0]["function"]["name"], "calculator")

  def test_get_mcp_schema(self):
    schemas = self.manager.get_mcp_schema
    self.assertLen(schemas, 1)
    self.assertEqual(schemas[0]["function"]["name"], "calculator")

  def test_register_mcp_tool(self):
    class NewTool(base_tool.BaseTool):

      @property
      def get_json_schema(self):
        return {"type": "function", "function": {"name": "new_tool"}}

    new_tool = NewTool(name="new_tool", description="A new tool.")
    self.manager.register_mcp_tool(new_tool)
    self.assertIn("new_tool", self.manager.names)

  def test_run_success(self):
    result = self.manager.run("calculator", a=10, b=5, op="+")
    self.assertEqual(result.output, "15")
    self.assertIsNone(result.error)

  def test_run_tool_not_found(self):
    result = self.manager.run("non_existent_tool")
    self.assertIsNotNone(result.error)
    self.assertIn("not registered", result.error)

  def test_run_tool_exception(self):
    mock_tool = mock.Mock(spec=base_tool.BaseTool)
    mock_tool.name = "buggy_tool"
    mock_tool.apply.side_effect = ValueError("Something broke")
    self.manager.register_mcp_tool(mock_tool)
    result = self.manager.run("buggy_tool", arg=1)
    self.assertIsNotNone(result.error)
