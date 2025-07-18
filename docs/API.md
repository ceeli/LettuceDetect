# LettuceDectect Web API

LettuceDetect comes with it's own web API and python client library. To use it, make sure to install the package with the optional API dependencies:

```bash
pip install -e .[api]
```

or

```bash
pip install lettucedetect[api]
```

Start the API server with the `scripts/start_api.py` script:

```bash
python scripts/start_api.py dev  # use "prod" for production environments
```

Usage:

```bash
usage: start_api.py [-h] [--model MODEL] [--method {transformer}] {prod,dev}

Start lettucedetect Web API.

positional arguments:
  {prod,dev}            Choose "dev" for development or "prod" for production
                        environments. The serve script uses "fastapi dev" for "dev" or
                        "fastapi run" for "prod" to start the web server. Additionally
                        when choosing the "dev" mode, python modules can be directly
                        imported from the repositroy without installing the package.

options:
  -h, --help            show this help message and exit
  --model MODEL         Path or huggingface URL to the model. The default value is
                        "KRLabsOrg/lettucedetect-base-modernbert-en-v1".
  --method {transformer}
                        Hallucination detection method. The default value is
                        "transformer".
```

Example using the python client library:

```python
from lettucedetect_api.client import LettuceClient

contexts = [
    "France is a country in Europe. "
    "The capital of France is Paris. "
    "The population of France is 67 million.",
]
question = "What is the capital of France? What is the population of France?"
answer = "The capital of France is Paris. The population of France is 69 million."

client = LettuceClient("http://127.0.0.1:8000")
response = client.detect_spans(contexts, question, answer)
print(response.predictions)

# [SpanDetectionItem(start=31, end=71, text=' The population of France is 69 million.', hallucination_score=0.989198625087738)]
```

See `demo/detection_api.ipynb` for more examples.
For async support use the `LettuceClientAsync` class instead.