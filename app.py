# Create a Bedrock Runtime client in the AWS Region of your choice.
import json
import boto3
from botocore.exceptions import ClientError


def invoke_mistral(
    prompt: str,
    region: str = "us-east-1",
    model_id: str = "mistral.mistral-large-2402-v1:0",
    max_tokens: int = 512,
    temperature: float = 0.5,
) -> str:
    """
    Invoke Mistral Large via AWS Bedrock and return the generated text.

    :param prompt: User prompt text
    :param region: AWS region where Bedrock is enabled
    :param model_id: Bedrock model ID
    :param max_tokens: Maximum tokens to generate
    :param temperature: Sampling temperature
    :return: Generated text response
    """

    client = boto3.client("bedrock-runtime", region_name=region)

    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    request_body = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
    except (ClientError, Exception) as e:
        raise RuntimeError(f"Failed to invoke model '{model_id}': {e}")

    model_response = json.loads(response["body"].read())
    return model_response["outputs"][0]["text"]
response=invoke_mistral(prompt="what is the best place of kolkata for a tourist?")
print(response)