# Multimodal Anomaly Detection with LLMs
In contrast to text-only large language models, multimodal large language models are designed to handle different forms of input, such as image and voice data. For timeseries data, vision-based approaches have shown strong potential compared to text-only representation. We leverage the ability of multimodal llms to process images and interpret the result in natural language to create an anomaly detection system.

We use a hybrid approach that combines isolation forest and markov anomaly detection algorithms. Isolation forests are good at spotting single-point deviations from the norm (e.g. the subject is in the kitchen in the middle of the night) while markov models are best for spotting unusual sequences (e.g. the subject goes directly to bed without eating). By creating numerical representations of the data, we can then convert our data into images which can be interpreted by the llm. Our approach is to create graphics of 24 hour periods, then check the most recent graph against the prior data. This approach is lightweight and processes large quantities of data rapidly, whereas an approach where the llm interacted with the data in tabular form would be much more intensive and prone to instability.

## Installing Requirements
### UV (Preferred)
Run `uv sync` to create a virtual environment with the required packages in the working directory.

### Pip
Create a virtual environment in the code directory. From inside the virtual environment, run `pip install .`

## Getting Started
Download and install Ollama, and use the CLI to download the desired llm model. We selected a low-parameter version of Qwen3-VL (qwen3-vl:2b). Fill out the `.env` file. Parameter count and model choice will depend on deployment.

```
# Example .env configuration
SENSOR_METADATA_PATH=./path/to/sensor/metadata.json
CSV_SENSOR_DATA_PATH=./path/to/sensor/metadata.csv
LLM_MODEL=qwen3-vl:2b
ANOMALY_DATA_PATH=./path/to/output.csv
GRAPH_OUTPUT_FOLDER=./path/to/graph/folder/
```
