# Multimodal Anomaly Detection with LLMs
LLMs have the ability to explain issues in natural language, but are poor at processing large quantities of data. Marking anomalies in advance to reduce the workload and then asking the algorithm to explain the anomaly in natural languages gives data scientists a starting point for tracing issues. 

In contrast to text-only large language models, multimodal large language models are designed to handle different forms of input, such as image and voice data. For timeseries data, vision-based approaches have shown strong potential compared to text-only representation.

By combining a relatively simple unsupervised learning algorithm with a multimodal large language model, we can create an anomaly detection system which helps reduce the odds of a false positive. The more rapid first detection pass reduces computational overhead or token usage, depending on how the llm is deployed.

We use a hybrid approach to anomaly detection that combines isolation forest and markov anomaly detection algorithms. Isolation forests are good at spotting single-point deviations from the norm (e.g. the subject is in the kitchen in the middle of the night) while markov models are best for spotting unusual sequences (e.g. the subject goes directly to bed without eating). 

## Getting Started
Download and install Ollama. Fill out the `.env` file. We selected a low-parameter version of Qwen3-VL (qwen3-vl:2b). Parameter count and model choice will depend on deployment.

```
# Example .env configuration
SENSOR_METADATA_PATH=./path/to/sensor/metadata.json
CSV_SENSOR_DATA_PATH=./path/to/sensor/metadata.csv
LLM_MODEL=qwen3-vl:2b
ANOMALY_DATA_PATH=./path/to/output.csv
```

## Ensuring consistent results
Ollama's structured output means that the results of each prompt will be sent in a consistent format between prompts, which can then be parsed by our script.
