# OpenAI API Fix

## Issue

The application was failing to deploy on Streamlit with the following error:

```
Error creating assistant: Error code: 400 - {'error': {'message': "Invalid value: 'retrieval'. Supported values are: 'code_interpreter', 'function', and 'file_search'.", 'type': 'invalid_request_error', 'param': 'tools[0].type', 'code': 'invalid_value'}}
```

This error occurred because the OpenAI Assistants API has been updated, and the tool type `"retrieval"` is no longer supported. The supported tool types are now:
- `"code_interpreter"`
- `"function"`
- `"file_search"`

## Fix

The fix was to update the `create_assistant_with_file` method in `openai_file_manager.py` to use the `"file_search"` tool type instead of `"retrieval"`:

```python
# Before
assistant = self.client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
    tools=[{"type": "retrieval"}]
)

# After
assistant = self.client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
    tools=[{"type": "file_search"}]  # Changed from "retrieval" to "file_search"
)
```

## Explanation

The OpenAI Assistants API has evolved, and the tool type for searching through files has been renamed from `"retrieval"` to `"file_search"`. This change is part of the API's ongoing development and improvement.

The fix ensures that the application uses the correct tool type when creating assistants, allowing the application to deploy and run successfully on Streamlit.

## References

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/assistants)
