from chainforge.providers import provider
from openai import OpenAI

# Initialize the OpenAI client pointed to LM Studio's local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# JSON schema for provider settings
LM_STUDIO_SETTINGS_SCHEMA = {
    "settings": {
        "temperature": {
            "type": "number",
            "title": "temperature",
            "description": "Controls randomness in the response generation",
            "default": 0.7,
            "minimum": 0,
            "maximum": 2.0,
            "multipleOf": 0.1,
        },
        "max_tokens": {
            "type": "integer",
            "title": "max_tokens",
            "description": "Maximum length of generated response",
            "default": 256,
            "minimum": 1,
            "maximum": 4096,
        },
        "system_prompt": {
            "type": "string",
            "title": "system_prompt",
            "description": "System prompt to control model behavior",
            "default": "You are a helpful AI assistant.",
        }
    },
    "ui": {
        "temperature": {
            "ui:help": "Higher values make output more random, lower values more deterministic",
            "ui:widget": "range"
        },
        "max_tokens": {
            "ui:help": "Maximum number of tokens in the generated response",
            "ui:widget": "range"
        },
        "system_prompt": {
            "ui:widget": "textarea"
        }
    }
}

@provider(
    name="LM Studio",
    emoji="🤖",
    models=["lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"],
    rate_limit="sequential",
    settings_schema=LM_STUDIO_SETTINGS_SCHEMA
)
def LMStudioCompletion(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
    system_prompt: str = "You are a helpful AI assistant.",
    **kwargs
) -> str:
    """
    Custom provider for LM Studio local models
    
    Args:
        prompt: The input prompt
        model: Model identifier
        temperature: Controls randomness (0-2)
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for chat completion
        **kwargs: Additional parameters passed to the completion call
    
    Returns:
        Generated text response
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LM Studio API: {str(e)}")
        return f"Error: {str(e)}"