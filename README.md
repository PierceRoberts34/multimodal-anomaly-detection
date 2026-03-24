# Interpreting Sensor Data with AI Agents
Using open source llms, we can interpret sensor data to create summaries. Careful prompt engineering leads to more consistent results, and rules like "answer in tabular format" allow us to capture and interpret output.

```
format = '|Activity|Start Time|End Time|Duration|Notes|'
system_prompt = f"""
  You are a data scientist tasked with interpreting home sensor data from sensors placed around a subject's house.
  Provide your answers in the following tabular format for easy parsing: {format}
"""
```

# Prerequisites
We need cleaned data to best make use of the agent. Cleaned data helps the AI produce more accurate summaries and improves stability

# Usage
Select a time window size. Larger time windows require more memory and risk information being ommitted in the summary, but smaller windows will take additional time to process since the prompt needs to be re-run every time.

# Current Results
Noise in the data seems to be confusing the llm. Specific training on sensor output may be needed to improve performance
