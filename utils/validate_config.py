import yaml
import json
from jsonschema import validate, RefResolver
import os

def load_and_validate_config(config_path, schema_path):
    """
    Loads a YAML configuration file and validates it against a JSON schema.

    Args:
        config_path (str): The path to the YAML config file.
        schema_path (str): The path to the master JSON schema file.

    Returns:
        dict: The validated configuration dictionary.

    Raises:
        jsonschema.ValidationError: If the configuration is invalid.
        FileNotFoundError: If the config or schema file cannot be found.
    """
    # --- Load Configuration ---
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Load Schema and Set up Resolver ---
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Create a resolver to handle the $ref to viability.schema.json
    # It needs to know the base URI from which to resolve relative paths.
    schema_dir = os.path.dirname(os.path.abspath(schema_path))
    resolver = RefResolver(base_uri=f'file://{schema_dir}/', referrer=schema)

    # --- Validate ---
    try:
        validate(instance=config, schema=schema, resolver=resolver)
        print("✅ Configuration is valid.")
        return config
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise e

if __name__ == '__main__':
    # Example usage:
    # This allows the script to be run directly for testing.
    try:
        load_and_validate_config('config.yaml', 'schemas/config.schema.json')
    except Exception:
        # Exit with a non-zero status code to indicate failure
        exit(1)