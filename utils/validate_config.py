import json
import yaml
import jsonschema
import os

def validate_config(config_path, schema_dir="schemas"):
    """
    Validates a YAML configuration file against a set of schemas.

    Args:
        config_path (str): Path to the YAML configuration file.
        schema_dir (str): Directory where schema files are stored.

    Returns:
        bool: True if validation succeeds, False otherwise.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return False

    # Validate the viability section
    viability_schema_path = os.path.join(schema_dir, "viability.schema.json")
    if "viability" in config and os.path.exists(viability_schema_path):
        with open(viability_schema_path, "r") as f:
            schema = json.load(f)

        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(config.get("viability")), key=str)

        if errors:
            print("Configuration validation failed for 'viability' section:")
            for error in errors:
                print(f"- {error.message} (path: {'/'.join(map(str, error.path))})")
            return False
        else:
            print("'viability' section validated successfully.")

    # Add other schema validations here as needed.

    print("Configuration validation passed.")
    return True

if __name__ == "__main__":
    # Example usage:
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print(f"Could not find {config_file}. Please run this script from the project root.")
    else:
        validate_config(config_file)