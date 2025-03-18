import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import json
import openai
import os

def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate statistics for each column in the dataframe."""
    stats = {}

    for column in df.columns:
        column_type = str(df[column].dtype)

        if np.issubdtype(df[column].dtype, np.number):
            # Numerical column statistics
            stats[column] = {
                'type': column_type,
                'stats': {
                    'mean': float(df[column].mean()),
                    'median': float(df[column].median()),
                    'std': float(df[column].std())
                }
            }
        else:
            # Categorical column statistics
            value_counts = df[column].value_counts().head(20)
            stats[column] = {
                'type': column_type,
                'distribution': {str(k): int(v) for k, v in value_counts.items()}
            }

    return stats

def prepare_gpt_prompt(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    """Prepare the prompt for GPT-4 analysis."""
    prompt = """Please analyze the following dataset and provide insights in a valid JSON format.

Dataset Overview:
- Total Rows: {rows}
- Total Columns: {columns}

Column Information:
{column_info}

Required JSON Response Format:
{{
    "dataset_description": "A comprehensive overview of the entire dataset",
    "suggested_analysis": [
        "Analysis suggestion 1",
        "Analysis suggestion 2",
        "Analysis suggestion 3"
    ],
    "columns": [
        {{
            "name": "column_name",
            "title": "Human readable title",
            "description": "Detailed description of the column and its significance",
            "confidence_score": 0.95
        }}
    ],
    "key_observations": [
        "Key observation 1",
        "Key observation 2",
        "Key observation 3"
    ]
}}

Note: For each column, provide a confidence score between 0 and 1 indicating how confident you are about the column title and description."""

    column_details = []
    for col, info in stats.items():
        if 'stats' in info:
            details = f"- {col} (Numerical):\n  Stats: {info['stats']}"
        else:
            details = f"- {col} (Categorical):\n  Distribution: {info['distribution']}"
        column_details.append(details)

    return prompt.format(
        rows=len(df),
        columns=len(df.columns),
        column_info="\n".join(column_details)
    )

def get_gpt_analysis(df: pd.DataFrame, stats: Dict[str, Any],
                    api_key: str, endpoint: str) -> Dict[str, Any]:
    """Get GPT-4 analysis of the dataset using Azure OpenAI.

    Parameters:
        df: DataFrame to analyze
        stats: Dictionary containing column statistics
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
    """
    # Configure Azure OpenAI settings
    openai.api_type = "azure"
    openai.api_base = endpoint
    openai.api_version = "2023-07-01-preview"
    openai.api_key = api_key

    prompt = prepare_gpt_prompt(df, stats)

    try:
        # Make API call
        response = openai.ChatCompletion.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis expert. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        # Parse the response and ensure all required fields exist
        analysis = json.loads(response.choices[0].message.content)
        return {
            "dataset_description": analysis.get("dataset_description", "No description available"),
            "suggested_analysis": analysis.get("suggested_analysis", []),
            "columns": analysis.get("columns", []),
            "key_observations": analysis.get("key_observations", [])
        }
    except json.JSONDecodeError as e:
        return {
            "dataset_description": "Error: Invalid JSON response from analysis",
            "suggested_analysis": ["Error: Could not generate analysis suggestions"],
            "columns": [],
            "key_observations": ["Error: Could not generate observations"]
        }
    except Exception as e:
        return {
            "dataset_description": f"Error: {str(e)}",
            "suggested_analysis": [],
            "columns": [],
            "key_observations": []
        }

def process_uploaded_file(uploaded_file, api_key: str, endpoint: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Process the uploaded CSV file and return dataframe, stats, and analysis."""
    df = pd.read_csv(uploaded_file)
    stats = get_column_stats(df)
    analysis = get_gpt_analysis(df, stats, api_key, endpoint)
    return df, stats, analysis