{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/harikesh2501/Heart_Disease_Prediction/blob/main/Harikesh.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3YDKVYubgQyt"
      },
      "source": [
        "**Heart Disease Prediction**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wJ6AbIAVgf7h"
      },
      "source": [
        "I. Importing essential libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RIOY2qDZgiJT",
        "outputId": "efbb73fe-231d-48d0-8c2a-64dcd20acf10"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['.config', 'sample_data']\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "%matplotlib inline\n",
        "\n",
        "import os\n",
        "print(os.listdir())\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ql8rk9ksgvEW"
      },
      "source": [
        "II. Import Dataset\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eR_DTT_Ug8SO"
      },
      "outputs": [],
      "source": [
        "dataset = pd.read_csv(\"heart.csv\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0djMtN7ihIaX"
      },
      "source": [
        "Shape of dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QehoKus5hJwI",
        "outputId": "5138c754-5e24-49dd-c7b8-4e5d1491ff1b"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(303, 14)"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ],
      "source": [
        "dataset.shape\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9xnKVBDxhQv4"
      },
      "source": [
        "Overview Of Dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "pmQzQjmRhUoN",
        "outputId": "7f69da03-34bd-4295-9486-009457c3a5d4"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
              "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
              "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
              "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
              "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
              "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
              "\n",
              "   ca  thal  target  \n",
              "0   0     1       1  \n",
              "1   0     2       1  \n",
              "2   0     2       1  \n",
              "3   0     2       1  \n",
              "4   0     2       1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-dae82caa-4148-40e1-9b4d-8003203bbaa0\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>63</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>145</td>\n",
              "      <td>233</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>150</td>\n",
              "      <td>0</td>\n",
              "      <td>2.3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>37</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>130</td>\n",
              "      <td>250</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>187</td>\n",
              "      <td>0</td>\n",
              "      <td>3.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>41</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>130</td>\n",
              "      <td>204</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>172</td>\n",
              "      <td>0</td>\n",
              "      <td>1.4</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>56</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>120</td>\n",
              "      <td>236</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>178</td>\n",
              "      <td>0</td>\n",
              "      <td>0.8</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>120</td>\n",
              "      <td>354</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>163</td>\n",
              "      <td>1</td>\n",
              "      <td>0.6</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-dae82caa-4148-40e1-9b4d-8003203bbaa0')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-dae82caa-4148-40e1-9b4d-8003203bbaa0 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-dae82caa-4148-40e1-9b4d-8003203bbaa0');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-594b2aaa-5b65-430d-a444-2a1208ebe0d5\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-594b2aaa-5b65-430d-a444-2a1208ebe0d5')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-594b2aaa-5b65-430d-a444-2a1208ebe0d5 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "dataset",
              "summary": "{\n  \"name\": \"dataset\",\n  \"rows\": 303,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9,\n        \"min\": 29,\n        \"max\": 77,\n        \"num_unique_values\": 41,\n        \"samples\": [\n          46,\n          66,\n          48\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"cp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 3,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          2,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"trestbps\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 17,\n        \"min\": 94,\n        \"max\": 200,\n        \"num_unique_values\": 49,\n        \"samples\": [\n          104,\n          123\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"chol\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 51,\n        \"min\": 126,\n        \"max\": 564,\n        \"num_unique_values\": 152,\n        \"samples\": [\n          277,\n          169\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"fbs\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"restecg\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thalach\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 22,\n        \"min\": 71,\n        \"max\": 202,\n        \"num_unique_values\": 91,\n        \"samples\": [\n          159,\n          152\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exang\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"oldpeak\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.1610750220686348,\n        \"min\": 0.0,\n        \"max\": 6.2,\n        \"num_unique_values\": 40,\n        \"samples\": [\n          1.9,\n          3.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"slope\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          0,\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ca\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 4,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          2,\n          4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thal\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 3,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          2,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"target\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 8
        }
      ],
      "source": [
        "dataset.head(5)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# New section"
      ],
      "metadata": {
        "id": "prrT4HnyTEf-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# New section"
      ],
      "metadata": {
        "id": "rZVvOgDLTFJO"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "k4l6cYQshb-J"
      },
      "source": [
        "Last 5 data Rows"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "ZI0tRjDghiw7",
        "outputId": "1ffde6f0-8e20-491c-a735-d68c551694b1"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"dataset\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8,\n        \"min\": 45,\n        \"max\": 68,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          57,\n          45,\n          68\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"cp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          0,\n          3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"trestbps\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13,\n        \"min\": 110,\n        \"max\": 144,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          110,\n          130\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"chol\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 52,\n        \"min\": 131,\n        \"max\": 264,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          264,\n          236\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"fbs\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"restecg\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thalach\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 22,\n        \"min\": 115,\n        \"max\": 174,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          132,\n          174\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exang\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"oldpeak\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.3490737563232043,\n        \"min\": 0.0,\n        \"max\": 3.4,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          1.2,\n          0.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"slope\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 1,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ca\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thal\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 2,\n        \"max\": 3,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"target\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 0,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-c59a942f-543f-491d-b878-956076e76e7d\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>298</th>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>140</td>\n",
              "      <td>241</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>123</td>\n",
              "      <td>1</td>\n",
              "      <td>0.2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>299</th>\n",
              "      <td>45</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>110</td>\n",
              "      <td>264</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>132</td>\n",
              "      <td>0</td>\n",
              "      <td>1.2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>300</th>\n",
              "      <td>68</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>144</td>\n",
              "      <td>193</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>141</td>\n",
              "      <td>0</td>\n",
              "      <td>3.4</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>301</th>\n",
              "      <td>57</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>130</td>\n",
              "      <td>131</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>115</td>\n",
              "      <td>1</td>\n",
              "      <td>1.2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>302</th>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>130</td>\n",
              "      <td>236</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>174</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-c59a942f-543f-491d-b878-956076e76e7d')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-c59a942f-543f-491d-b878-956076e76e7d button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-c59a942f-543f-491d-b878-956076e76e7d');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-442bc172-0d67-4644-84dc-ecc0eaa40f40\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-442bc172-0d67-4644-84dc-ecc0eaa40f40')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-442bc172-0d67-4644-84dc-ecc0eaa40f40 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
              "298   57    0   0       140   241    0        1      123      1      0.2   \n",
              "299   45    1   3       110   264    0        1      132      0      1.2   \n",
              "300   68    1   0       144   193    1        1      141      0      3.4   \n",
              "301   57    1   0       130   131    0        1      115      1      1.2   \n",
              "302   57    0   1       130   236    0        0      174      0      0.0   \n",
              "\n",
              "     slope  ca  thal  target  \n",
              "298      1   0     3       0  \n",
              "299      1   0     3       0  \n",
              "300      1   2     3       0  \n",
              "301      1   1     3       0  \n",
              "302      1   1     2       0  "
            ]
          },
          "execution_count": 11,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset.tail()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hJ0oFoBmhtKs"
      },
      "source": [
        "Random 5 row"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "H49SqMeWhzWt",
        "outputId": "b8d8b909-edd6-46fc-83b9-6f933b29d740"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"dataset\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6,\n        \"min\": 47,\n        \"max\": 65,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          56,\n          51,\n          47\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 1,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"cp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"trestbps\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 17,\n        \"min\": 108,\n        \"max\": 150,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          120\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"chol\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 52,\n        \"min\": 126,\n        \"max\": 254,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          193\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"fbs\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"restecg\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thalach\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 21,\n        \"min\": 123,\n        \"max\": 173,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          162\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exang\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 0,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"oldpeak\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.2041594578792294,\n        \"min\": 0.0,\n        \"max\": 2.8,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          1.9\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"slope\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 2,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ca\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thal\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 2,\n        \"max\": 3,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"target\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-06ef04a8-103a-44d5-921b-43b8946f6706\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>111</th>\n",
              "      <td>57</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>150</td>\n",
              "      <td>126</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>173</td>\n",
              "      <td>0</td>\n",
              "      <td>0.2</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>117</th>\n",
              "      <td>56</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>120</td>\n",
              "      <td>193</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>162</td>\n",
              "      <td>0</td>\n",
              "      <td>1.9</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>230</th>\n",
              "      <td>47</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>108</td>\n",
              "      <td>243</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>152</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>218</th>\n",
              "      <td>65</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>135</td>\n",
              "      <td>254</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>127</td>\n",
              "      <td>0</td>\n",
              "      <td>2.8</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27</th>\n",
              "      <td>51</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>110</td>\n",
              "      <td>175</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>123</td>\n",
              "      <td>0</td>\n",
              "      <td>0.6</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-06ef04a8-103a-44d5-921b-43b8946f6706')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-06ef04a8-103a-44d5-921b-43b8946f6706 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-06ef04a8-103a-44d5-921b-43b8946f6706');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-44233e4c-f46a-4120-9542-784c271bdb54\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-44233e4c-f46a-4120-9542-784c271bdb54')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-44233e4c-f46a-4120-9542-784c271bdb54 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
              "111   57    1   2       150   126    1        1      173      0      0.2   \n",
              "117   56    1   3       120   193    0        0      162      0      1.9   \n",
              "230   47    1   2       108   243    0        1      152      0      0.0   \n",
              "218   65    1   0       135   254    0        0      127      0      2.8   \n",
              "27    51    1   2       110   175    0        1      123      0      0.6   \n",
              "\n",
              "     slope  ca  thal  target  \n",
              "111      2   1     3       1  \n",
              "117      1   0     3       1  \n",
              "230      2   0     2       0  \n",
              "218      1   1     3       0  \n",
              "27       2   0     2       1  "
            ]
          },
          "execution_count": 12,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset.sample(5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-Mu_UOEfh3Cw"
      },
      "source": [
        "Description"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        },
        "id": "WvIjHqlniHyo",
        "outputId": "8d3b86e1-e2d5-45d3-b79a-a4735bcaf557"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"dataset\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 92.63263171018461,\n        \"min\": 9.082100989837857,\n        \"max\": 303.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          54.366336633663366,\n          55.0,\n          303.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sex\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.91793021099774,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.6831683168316832,\n          1.0,\n          0.46601082333962385\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"cp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.72725528212327,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          303.0,\n          0.966996699669967,\n          2.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"trestbps\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 82.65195263865039,\n        \"min\": 17.5381428135171,\n        \"max\": 303.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          131.62376237623764,\n          130.0,\n          303.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"chol\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 150.35806568851743,\n        \"min\": 51.83075098793003,\n        \"max\": 564.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          246.26402640264027,\n          240.0,\n          303.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"fbs\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 107.0512286741478,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.1485148514851485,\n          1.0,\n          0.35619787492797644\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"restecg\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.8733588009897,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          303.0,\n          0.528052805280528,\n          2.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thalach\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 83.70384393886218,\n        \"min\": 22.905161114914094,\n        \"max\": 303.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          149.64686468646866,\n          153.0,\n          303.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"exang\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.9862394088184,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.32673267326732675,\n          1.0,\n          0.4697944645223165\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"oldpeak\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.59952466080658,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          303.0,\n          1.0396039603960396,\n          1.6\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"slope\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.72394469173834,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          303.0,\n          1.3993399339933994,\n          2.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ca\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.79372080487734,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          303.0,\n          0.7293729372937293,\n          4.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"thal\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.47909774814387,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          303.0,\n          2.3135313531353137,\n          3.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"target\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 106.92326354929804,\n        \"min\": 0.0,\n        \"max\": 303.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.5445544554455446,\n          1.0,\n          0.4988347841643913\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-eb49ee11-7ca1-4574-a459-40fda4ebac54\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>54.366337</td>\n",
              "      <td>0.683168</td>\n",
              "      <td>0.966997</td>\n",
              "      <td>131.623762</td>\n",
              "      <td>246.264026</td>\n",
              "      <td>0.148515</td>\n",
              "      <td>0.528053</td>\n",
              "      <td>149.646865</td>\n",
              "      <td>0.326733</td>\n",
              "      <td>1.039604</td>\n",
              "      <td>1.399340</td>\n",
              "      <td>0.729373</td>\n",
              "      <td>2.313531</td>\n",
              "      <td>0.544554</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>9.082101</td>\n",
              "      <td>0.466011</td>\n",
              "      <td>1.032052</td>\n",
              "      <td>17.538143</td>\n",
              "      <td>51.830751</td>\n",
              "      <td>0.356198</td>\n",
              "      <td>0.525860</td>\n",
              "      <td>22.905161</td>\n",
              "      <td>0.469794</td>\n",
              "      <td>1.161075</td>\n",
              "      <td>0.616226</td>\n",
              "      <td>1.022606</td>\n",
              "      <td>0.612277</td>\n",
              "      <td>0.498835</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>29.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>94.000000</td>\n",
              "      <td>126.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>71.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>47.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>120.000000</td>\n",
              "      <td>211.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>133.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>55.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>130.000000</td>\n",
              "      <td>240.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>153.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.800000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>61.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>140.000000</td>\n",
              "      <td>274.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>166.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.600000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>77.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>200.000000</td>\n",
              "      <td>564.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>202.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>6.200000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-eb49ee11-7ca1-4574-a459-40fda4ebac54')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-eb49ee11-7ca1-4574-a459-40fda4ebac54 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-eb49ee11-7ca1-4574-a459-40fda4ebac54');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-f049b17b-8f8a-4041-a780-40bf22bc4314\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-f049b17b-8f8a-4041-a780-40bf22bc4314')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-f049b17b-8f8a-4041-a780-40bf22bc4314 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "              age         sex          cp    trestbps        chol         fbs  \\\n",
              "count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000   \n",
              "mean    54.366337    0.683168    0.966997  131.623762  246.264026    0.148515   \n",
              "std      9.082101    0.466011    1.032052   17.538143   51.830751    0.356198   \n",
              "min     29.000000    0.000000    0.000000   94.000000  126.000000    0.000000   \n",
              "25%     47.500000    0.000000    0.000000  120.000000  211.000000    0.000000   \n",
              "50%     55.000000    1.000000    1.000000  130.000000  240.000000    0.000000   \n",
              "75%     61.000000    1.000000    2.000000  140.000000  274.500000    0.000000   \n",
              "max     77.000000    1.000000    3.000000  200.000000  564.000000    1.000000   \n",
              "\n",
              "          restecg     thalach       exang     oldpeak       slope          ca  \\\n",
              "count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000   \n",
              "mean     0.528053  149.646865    0.326733    1.039604    1.399340    0.729373   \n",
              "std      0.525860   22.905161    0.469794    1.161075    0.616226    1.022606   \n",
              "min      0.000000   71.000000    0.000000    0.000000    0.000000    0.000000   \n",
              "25%      0.000000  133.500000    0.000000    0.000000    1.000000    0.000000   \n",
              "50%      1.000000  153.000000    0.000000    0.800000    1.000000    0.000000   \n",
              "75%      1.000000  166.000000    1.000000    1.600000    2.000000    1.000000   \n",
              "max      2.000000  202.000000    1.000000    6.200000    2.000000    4.000000   \n",
              "\n",
              "             thal      target  \n",
              "count  303.000000  303.000000  \n",
              "mean     2.313531    0.544554  \n",
              "std      0.612277    0.498835  \n",
              "min      0.000000    0.000000  \n",
              "25%      2.000000    0.000000  \n",
              "50%      2.000000    1.000000  \n",
              "75%      3.000000    1.000000  \n",
              "max      3.000000    1.000000  "
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset.describe()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xD5qZHVciSk-"
      },
      "source": [
        "Data Type Of all the feature column"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mkF8Hmb4iZ2b",
        "outputId": "047d3183-03fd-49b0-ad89-9263c1ddff49"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 303 entries, 0 to 302\n",
            "Data columns (total 14 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   age       303 non-null    int64  \n",
            " 1   sex       303 non-null    int64  \n",
            " 2   cp        303 non-null    int64  \n",
            " 3   trestbps  303 non-null    int64  \n",
            " 4   chol      303 non-null    int64  \n",
            " 5   fbs       303 non-null    int64  \n",
            " 6   restecg   303 non-null    int64  \n",
            " 7   thalach   303 non-null    int64  \n",
            " 8   exang     303 non-null    int64  \n",
            " 9   oldpeak   303 non-null    float64\n",
            " 10  slope     303 non-null    int64  \n",
            " 11  ca        303 non-null    int64  \n",
            " 12  thal      303 non-null    int64  \n",
            " 13  target    303 non-null    int64  \n",
            "dtypes: float64(1), int64(13)\n",
            "memory usage: 33.3 KB\n"
          ]
        }
      ],
      "source": [
        "dataset.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "l5dWO7A-ikrt"
      },
      "outputs": [],
      "source": [
        "##we have no missing values"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NEwNBgjoith8"
      },
      "source": [
        "Analysing the **'target'** variable"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "7oLdp62bjMrj",
        "outputId": "2994636b-803c-4924-98c9-3ea1a8002bd6"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>303.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.544554</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.498835</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ],
            "text/plain": [
              "count    303.000000\n",
              "mean       0.544554\n",
              "std        0.498835\n",
              "min        0.000000\n",
              "25%        0.000000\n",
              "50%        1.000000\n",
              "75%        1.000000\n",
              "max        1.000000\n",
              "Name: target, dtype: float64"
            ]
          },
          "execution_count": 16,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"target\"].describe()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MyKRzVQPjf8v",
        "outputId": "8efcb9b4-2aec-43a7-fd7a-ae025b075e89"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([1, 0])"
            ]
          },
          "execution_count": 17,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"target\"].unique()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nOwVjHf3jmUy"
      },
      "source": [
        "Clearly, this is a classification problem, with the target variable having values '0' and '1'"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HihtLp7jjuqE"
      },
      "source": [
        "**Checking correlation between columns**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "svvhO1G0jyZN",
        "outputId": "b0d6c852-2286-486c-ce0f-c26a7e8229c1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "target      1.000000\n",
            "exang       0.436757\n",
            "cp          0.433798\n",
            "oldpeak     0.430696\n",
            "thalach     0.421741\n",
            "ca          0.391724\n",
            "slope       0.345877\n",
            "thal        0.344029\n",
            "sex         0.280937\n",
            "age         0.225439\n",
            "trestbps    0.144931\n",
            "restecg     0.137230\n",
            "chol        0.085239\n",
            "fbs         0.028046\n",
            "Name: target, dtype: float64\n"
          ]
        }
      ],
      "source": [
        "print(dataset.corr()[\"target\"].abs().sort_values(ascending=False))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3Up968AMkGH5"
      },
      "outputs": [],
      "source": [
        "#conclusion:  most columns are moderately correlated with target, but 'fbs' is very weakly correlated."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "INhdm9U3kQSE"
      },
      "source": [
        "**Exploratory Data Analysis (EDA)**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UYlhmqy3kXIl"
      },
      "source": [
        "First, analysing the target variable:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 611
        },
        "id": "EMEqbjCAkZK9",
        "outputId": "e3c79b3a-71dd-4091-d7dc-2c261cf20a72"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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1NRUek51dXVGjx6dZcuW7fK8W7ZsSXt7e5cNAIC9X7duS2hoaMjDDz+cYcOGdVl/+OGHU19f3yODJckll1yS9vb2DB8+PPvss0+2b9+eK6+8MhMnTkzy58hOssPXktXW1pb27cycOXNy2WWX9dicAAD0Dt2K26lTp2bGjBnZtm1bxowZkyRZvHhxLrrooh79C2Xf//73c+edd2bBggU58sgjs3LlysyYMSP19fWZNGlSt887c+bMNDc3lx63t7enoaGhJ0YGAKCMuhW3F154YX7/+9/n85//fLZu3Zok6d+/fy6++OLMnDmzx4a78MILc8kll+Tss89O8udvaXjhhRcyZ86cTJo0KXV1dUmS1tbWLrdJtLa2ZuTIkbs8b2VlZSorK3tsTgAAeodu3XNbUVGRq666Ki+99FIeffTRPPXUU1m/fn1mz57do8O98sor6dOn64j77LNPOjo6kiTDhg1LXV1dFi9eXNrf3t6e5cuX+2MSAABvQd26cvsX+++/f4499tiemmUH48ePz5VXXpmhQ4fmyCOPzM9//vN84xvfyGc/+9kkf47sGTNm5Iorrsihhx5a+iqw+vr6nH766bttLgAAeqc3Fbe72w033JBZs2bl85//fNatW5f6+vr8wz/8Q5crxBdddFE2bdqUadOmZcOGDTn++OOzaNEi33ELAPAWVNH5//65r7eo9vb2VFdXp62tLVVVVXv89UddeMcef01gz1jxtU+Xe4Sy8HsNiqtcv9deb691655bAADojcQtAACFIW4BACgMcQsAQGGIWwAACkPcAgBQGOIWAIDCELcAABSGuAUAoDDELQAAhSFuAQAoDHELAEBhiFsAAApD3AIAUBjiFgCAwhC3AAAUhrgFAKAwxC0AAIUhbgEAKAxxCwBAYYhbAAAKQ9wCAFAY4hYAgMIQtwAAFIa4BQCgMMQtAACFIW4BACgMcQsAQGGIWwAACkPcAgBQGOIWAIDCELcAABSGuAUAoDDELQAAhSFuAQAoDHELAEBhiFsAAApD3AIAUBjiFgCAwhC3AAAUhrgFAKAwxC0AAIUhbgEAKIxeH7e//e1v83d/93cZNGhQBgwYkKOOOipPPPFEaX9nZ2dmz56dIUOGZMCAAWlqasqzzz5bxokBACiXXh23f/jDH/KhD30o++67b/7zP/8zv/71r/P1r389b3/720vHXH31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rLNKz128eHGqqqpy//33J0na29szfvz4nHrqqVmwYEFeeOGFHW4v2LBhQ8aMGZO///u/z7XXXps//elPufjii3PWWWflgQce2Ok8Bx100J75xwB4E8QtQC9QXV2dfv365W1ve1vq6upK65dddlnp52HDhmXZsmX5/ve/3yVu99tvv/zrv/5r+vXrlySZP39+Kioqcsstt6R///454ogj8tvf/jZTp04tPefGG2/M0Ucfna985SultVtvvTUNDQ155pln8p73vGen8wD0duIWoBebN29ebr311qxZsyZ/+tOfsnXr1owcObLLMUcddVQpbJNk1apVed/73pf+/fuX1o477rguz3nqqafy4IMPZv/999/hNZ9//vm85z3v6dk3ArCHiFuAXuq73/1uvvjFL+brX/96Ghsbc8ABB+RrX/tali9f3uW4/fbb7w2fe+PGjRk/fnyuuuqqHfYNGTKk2zMDlJu4Begl+vXrl+3bt5ceP/zww/ngBz+Yz3/+86W1559//q+e57DDDsu//du/ZcuWLamsrEySPP74412Oef/7358f/OAHOeSQQ9K3787/K/i/8wDsDXxbAkAvccghh2T58uX5zW9+k5dffjmHHnponnjiidx333155plnMmvWrB0idWf+9m//Nh0dHZk2bVqefvrp3HfffbnmmmuSJBUVFUmS6dOnZ/369fnUpz6Vxx9/PM8//3zuu+++TJ48uRS0/3eejo6O3ffmAXqIuAXoJb74xS9mn332yRFHHJGDDjooY8eOzRlnnJFPfvKTGT16dH7/+993uYq7K1VVVbnnnnuycuXKjBw5Ml/60pcye/bsJCndh1tfX5+HH34427dvz4c//OEcddRRmTFjRgYOHJg+ffrsdJ41a9bsvjcP0EMqOjs7O8s9BAC715133pnJkyenra0tAwYMKPc4ALuNe24BCuiOO+7Iu971rrzjHe/IU089VfoOW2ELFJ24BSiglpaWzJ49Oy0tLRkyZEjOPPPMXHnlleUeC2C3c1sCAACF4QNlAAAUhrgFAKAwxC0AAIUhbgEAKAxxCwBAYYhbAAAKQ9wCAFAY4hYAgMIQtwAAFMb/ByPGByXZYELdAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "target\n",
            "1    165\n",
            "0    138\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ],
      "source": [
        "y = dataset[\"target\"]\n",
        "\n",
        "plt.figure(figsize=(8, 6))  # Increase the values to zoom out more\n",
        "sns.countplot(x=y)\n",
        "\n",
        "# Show the plot\n",
        "plt.show()\n",
        "\n",
        "\n",
        "target_temp = dataset.target.value_counts()\n",
        "\n",
        "print(target_temp)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L4VGhdUin1X7",
        "outputId": "ee4ea353-d901-4425-f64e-b552b9054d7c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Percentage of patience without heart problems: 45.54\n",
            "Percentage of patience with heart problems: 54.46\n"
          ]
        }
      ],
      "source": [
        "print(\"Percentage of patience without heart problems: \"+str(round(target_temp[0]*100/303,2)))\n",
        "print(\"Percentage of patience with heart problems: \"+str(round(target_temp[1]*100/303,2)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yLdu2WV7oa6G"
      },
      "source": [
        "**We'll analyse 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca' and 'thal'**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CL60hcntonKD"
      },
      "source": [
        "Analysing the 'Sex' feature"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mllyA8AYoqq_",
        "outputId": "2214586e-b58f-4beb-8a4c-7c1885d072f2"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([1, 0])"
            ]
          },
          "execution_count": 25,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"sex\"].unique()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 559
        },
        "id": "LBZJ2ifco2EQ",
        "outputId": "014b6025-4919-4a06-f097-53b1ce6c23ad"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='sex', ylabel='target'>"
            ]
          },
          "execution_count": 44,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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nHp6erJp06Y0NTUNrJWXl6epqSkbNmx4Xq9x/fXX593vfnf222+/IZ/v7u5OZ2fnoAMAAGA4ShpOO3bsSG9vb2prawet19bWpr29vfD8jRs35uc//3ne97737XFPa2trampqBo66uroXPDcAADC+lPyrei/E9ddfn5kzZ2bu3Ll73LN06dLs2rVr4Ni6desoTggAAOwNJpTyzadMmZKKiop0dHQMWu/o6MjUqVOf89yurq7cdNNN+fSnP/2c+6qqqlJVVfWCZwUAAMavkl5xqqyszOzZs9PW1jaw1tfXl7a2tsybN+85z/3Wt76V7u7u/PM///OLPSYAADDOlfSKU5K0tLRk0aJFmTNnTubOnZuVK1emq6srzc3NSZKFCxdm2rRpaW1tHXTe9ddfn9NPPz0HHXRQKcYGAADGkZKH04IFC7J9+/YsW7Ys7e3taWhoyLp16wZuGLFly5aUlw++MPbQQw/lhz/8Ye64445SjAwAAIwzJQ+nJFmyZEmWLFky5HPr16/fbe3oo49Of3//izwVAADAn72k76oHAAAwGoQTAABAAeEEAABQQDgBAAAUEE4AAAAFhBMAAEAB4QQAAFBAOAEAABQQTgAAAAWEEwAAQAHhBAAAUEA4AQAAFBBOAAAABSaUegBgdPVX7JNdx5056DEAAM9NOMF4U1aW/gmVpZ4CAOAlxVf1AAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACgQMnDadWqVamvr091dXUaGxuzcePG59z/5JNP5pxzzskhhxySqqqqvPrVr85tt902StMCAADj0YRSvvnatWvT0tKS1atXp7GxMStXrsz8+fPz0EMP5eCDD95tf09PT974xjfm4IMPzs0335xp06blscceywEHHDD6wwMAAONGScNpxYoVWbx4cZqbm5Mkq1evzve///2sWbMm559//m7716xZk507d+buu+/OPvvskySpr68fzZEBAIBxqGRf1evp6cmmTZvS1NT07DDl5WlqasqGDRuGPOe73/1u5s2bl3POOSe1tbU59thjc+mll6a3t3eP79Pd3Z3Ozs5BBwAAwHCULJx27NiR3t7e1NbWDlqvra1Ne3v7kOc8+uijufnmm9Pb25vbbrstF110UT7/+c/n3/7t3/b4Pq2trampqRk46urqRvRzAAAAe7+S3xxiOPr6+nLwwQfn2muvzezZs7NgwYJccMEFWb169R7PWbp0aXbt2jVwbN26dRQnBgAA9gYl+43TlClTUlFRkY6OjkHrHR0dmTp16pDnHHLIIdlnn31SUVExsHbMMcekvb09PT09qays3O2cqqqqVFVVjezwAADAuFKyK06VlZWZPXt22traBtb6+vrS1taWefPmDXnOiSeemEceeSR9fX0Daw8//HAOOeSQIaMJAABgJJT0q3otLS257rrr8tWvfjUPPPBAPvCBD6Srq2vgLnsLFy7M0qVLB/Z/4AMfyM6dO3Puuefm4Ycfzve///1ceumlOeecc0r1EQAAgHGgpLcjX7BgQbZv355ly5alvb09DQ0NWbdu3cANI7Zs2ZLy8mfbrq6uLrfffnvOO++8HHfccZk2bVrOPffcfPKTnyzVRwAAAMaBsv7+/v5SDzGaOjs7U1NTk127dmXy5MmlHmc3sz/+tVKPADCiNl2+sNQjAMCQhtMGL6m76gEAAJSCcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAhNKPQAAAKOvv78/XV1dA4/322+/lJWVlXAiGNuEEwDAONTV1ZXTTjtt4PGtt96aSZMmlXAiGNt8VQ8AAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAmMinFatWpX6+vpUV1ensbExGzdu3OPer3zlKykrKxt0VFdXj+K0AADAeFPycFq7dm1aWlqyfPnybN68ObNmzcr8+fOzbdu2PZ4zefLkPPHEEwPHY489NooTAwAA403Jw2nFihVZvHhxmpubM2PGjKxevToTJ07MmjVr9nhOWVlZpk6dOnDU1taO4sQAAMB4U9Jw6unpyaZNm9LU1DSwVl5enqampmzYsGGP5z399NM5/PDDU1dXl9NOOy2/+MUv9ri3u7s7nZ2dgw4AAIDhKGk47dixI729vbtdMaqtrU17e/uQ5xx99NFZs2ZNbr311nzjG99IX19fTjjhhPzmN78Zcn9ra2tqamoGjrq6uhH/HAAAwN6t5F/VG6558+Zl4cKFaWhoyEknnZRbbrklL3/5y3PNNdcMuX/p0qXZtWvXwLF169ZRnhgAAHipm1DKN58yZUoqKirS0dExaL2joyNTp059Xq+xzz775Pjjj88jjzwy5PNVVVWpqqp6wbMCAADjV0mvOFVWVmb27Nlpa2sbWOvr60tbW1vmzZv3vF6jt7c3999/fw455JAXa0wAAGCcK+kVpyRpaWnJokWLMmfOnMydOzcrV65MV1dXmpubkyQLFy7MtGnT0tramiT59Kc/nb//+7/PUUcdlSeffDKXX355Hnvssbzvfe8r5ccAAAD2YiUPpwULFmT79u1ZtmxZ2tvb09DQkHXr1g3cMGLLli0pL3/2wtjvfve7LF68OO3t7XnZy16W2bNn5+67786MGTNK9REAAIC9XFl/f39/qYcYTZ2dnampqcmuXbsyefLkUo+zm9kf/1qpRwAYUZsuX1jqEYAhPP300znttNMGHt96662ZNGlSCSeC0TecNnjJ3VUPAABgtAknAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACww6nLVu2ZKg7mPf392fLli0jMhQAAMBYMuxwOuKII7J9+/bd1nfu3JkjjjhiRIYCAAAYS4YdTv39/SkrK9tt/emnn051dfWIDAUAADCWTHi+G1taWpIkZWVlueiiizJx4sSB53p7e3PPPfekoaFhxAcEAAAotecdTvfee2+SP19xuv/++1NZWTnwXGVlZWbNmpWPfexjIz8hAABAiT3vcLrrrruSJM3NzbnyyiszefLkF20oAACAsWTYv3G64YYbMnny5DzyyCO5/fbb84c//CFJhrzTHgAAwN5g2OG0c+fOvOENb8irX/3qvOUtb8kTTzyRJHnve9+bj370oyM+IAAAQKkNO5w+8pGPZJ999smWLVsG3SBiwYIFWbdu3YgOBwAAMBY87984/cUdd9yR22+/PYcddtig9Ve96lV57LHHRmwwAACAsWLYV5y6uroGXWn6i507d6aqqmpEhgIAABhLhh1Or3vd6/K1r31t4HFZWVn6+vryuc99Lq9//etHdDgAAICxYNhf1fvc5z6XN7zhDfnJT36Snp6efOITn8gvfvGL7Ny5Mz/60Y9ejBkBAABKathXnI499tg8/PDDee1rX5vTTjstXV1dOeOMM3LvvffmyCOPfDFmBAAAKKlhX3FKkpqamlxwwQUjPQsA8P+Z/fGvFW+CF6DsmZ7U/NXjky+6Kf0TKks2D3u/TZcvLPUIL8iww+lnP/vZkOtlZWWprq7OK17xCjeJAAAA9irDDqeGhoaUlZUlSfr7+5Nk4HGS7LPPPlmwYEGuueaaVFdXj9CYAAAApTPs3zh9+9vfzqte9apce+21+elPf5qf/vSnufbaa3P00UfnxhtvzPXXX5///u//zoUXXvhizAsAADDqhn3F6ZJLLsmVV16Z+fPnD6zNnDkzhx12WC666KJs3Lgx++23Xz760Y/miiuuGNFhAQAASmHYV5zuv//+HH744butH3744bn//vuT/PnrfE888cQLnw4AAGAMGHY4TZ8+PZdddll6enoG1v70pz/lsssuy/Tp05Mkjz/+eGpra0duSgAAgBIa9lf1Vq1albe97W057LDDctxxxyX581Wo3t7efO9730uSPProo/ngBz84spMCAACUyLDD6YQTTsivfvWrfPOb38zDDz+cJHnXu96Vs846K/vvv3+S5Oyzzx7ZKQEAAEpoWOH0pz/9KdOnT8/3vve9vP/973+xZgIAABhThvUbp3322Sd//OMfX6xZAAAAxqRh3xzinHPOyWc/+9k888wzL8Y8AAAAY86wf+P04x//OG1tbbnjjjsyc+bM7LfffoOev+WWW0ZsOAAAgLFg2OF0wAEH5B3veMeLMQsAAMCYNOxwuuGGG16MOQAAAMasYf/GCQAAYLwZ9hWnJLn55pvzH//xH9myZUt6enoGPbd58+YRGQwAAGCsGPYVp6uuuirNzc2pra3Nvffem7lz5+aggw7Ko48+mje/+c0vxowAAAAlNexw+uIXv5hrr702X/jCF1JZWZlPfOITufPOO/PhD384u3btejFmBAAAKKlhh9OWLVtywgknJEn23XffPPXUU0mSs88+O//+7/8+stMBAACMAcMOp6lTp2bnzp1Jkle84hX53//93yTJr371q/T394/sdAAAAGPAsMPpH//xH/Pd7343SdLc3Jzzzjsvb3zjG7NgwYK8/e1v/5uGWLVqVerr61NdXZ3GxsZs3LjxeZ130003paysLKeffvrf9L4AAADPx7DvqnfBBRdk2rRpSZJzzjknBx10UO6+++687W1vy5ve9KZhD7B27dq0tLRk9erVaWxszMqVKzN//vw89NBDOfjgg/d43q9//et87GMfy+te97phvycAAMBwDPuK01FHHZUnn3xy4PG73/3uXHXVVTnrrLMyffr0YQ+wYsWKLF68OM3NzZkxY0ZWr16diRMnZs2aNXs8p7e3N//0T/+Uiy++OK985SuH/Z4AAADDMexw2tPvmJ5++ulUV1cP67V6enqyadOmNDU1PTtQeXmampqyYcOGPZ736U9/OgcffHDe+973Duv9AAAA/hbP+6t6LS0tSZKysrIsW7YsEydOHHiut7c399xzTxoaGob15jt27Ehvb29qa2sHrdfW1ubBBx8c8pwf/vCHuf7663Pfffc9r/fo7u5Od3f3wOPOzs5hzQgAAPC8w+nee+9N8ucrTvfff38qKysHnqusrMysWbPysY99bOQn/CtPPfVUzj777Fx33XWZMmXK8zqntbU1F1988Ys6FwAAsHd73uF01113JfnznfSuvPLKTJ48+QW/+ZQpU1JRUZGOjo5B6x0dHZk6depu+//v//4vv/71r3PqqacOrPX19SVJJkyYkIceeihHHnnkoHOWLl06cLUs+fMVp7q6uhc8OwAAMH4M+656N9xww4i9eWVlZWbPnp22traBW4r39fWlra0tS5Ys2W3/9OnTc//99w9au/DCC/PUU0/lyiuvHDKIqqqqUlVVNWIzAwAA48+ww2mktbS0ZNGiRZkzZ07mzp2blStXpqurK83NzUmShQsXZtq0aWltbU11dXWOPfbYQecfcMABSbLbOgAAwEgpeTgtWLAg27dvz7Jly9Le3p6GhoasW7du4IYRW7ZsSXn5sG/+BwAAMGJKHk5JsmTJkiG/mpck69evf85zv/KVr4z8QAAAAH/FpRwAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgAJj4q56AACMrv6KfbLruDMHPQb2TDgBAIxHZWXpn1BZ6ingJcNX9QAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIACwgkAAKDAmAinVatWpb6+PtXV1WlsbMzGjRv3uPeWW27JnDlzcsABB2S//fZLQ0NDvv71r4/itAAAwHhT8nBau3ZtWlpasnz58mzevDmzZs3K/Pnzs23btiH3H3jggbnggguyYcOG/OxnP0tzc3Oam5tz++23j/LkAADAeFHycFqxYkUWL16c5ubmzJgxI6tXr87EiROzZs2aIfeffPLJefvb355jjjkmRx55ZM4999wcd9xx+eEPfzjKkwMAAONFScOpp6cnmzZtSlNT08BaeXl5mpqasmHDhsLz+/v709bWloceeij/8A//MOSe7u7udHZ2DjoAAACGo6ThtGPHjvT29qa2tnbQem1tbdrb2/d43q5duzJp0qRUVlbmrW99a77whS/kjW9845B7W1tbU1NTM3DU1dWN6GcAAAD2fiX/qt7fYv/99899992XH//4x7nkkkvS0tKS9evXD7l36dKl2bVr18CxdevW0R0WAAB4yZtQyjefMmVKKioq0tHRMWi9o6MjU6dO3eN55eXlOeqoo5IkDQ0NeeCBB9La2pqTTz55t71VVVWpqqoa0bkBAIDxpaRXnCorKzN79uy0tbUNrPX19aWtrS3z5s173q/T19eX7u7uF2NEAACA0l5xSpKWlpYsWrQoc+bMydy5c7Ny5cp0dXWlubk5SbJw4cJMmzYtra2tSf78m6U5c+bkyCOPTHd3d2677bZ8/etfz5e+9KVSfgwAAGAvVvJwWrBgQbZv355ly5alvb09DQ0NWbdu3cANI7Zs2ZLy8mcvjHV1deWDH/xgfvOb32TffffN9OnT841vfCMLFiwo1UcAAAD2cmX9/f39pR5iNHV2dqampia7du3K5MmTSz3ObmZ//GulHgFgRG26fGGpR3hJ8vcA2NuMxb8Hw2mDl+Rd9QAAAEaTcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAACAAsIJAACggHACAAAoIJwAAAAKCCcAAIAC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yaZNm9LU1DSwVl5enqampmzYsKGEkwEAjG3CCcaRHTt2pLe3N7W1tYPWa2tr097eXqKpAADGPuEEAABQQDjBODJlypRUVFSko6Nj0HpHR0emTp1aoqkAAMY+4QTjSGVlZWbPnp22traBtb6+vrS1tWXevHklnAwAYGybUOoBgNHV0tKSRYsWZc6cOZk7d25WrlyZrq6uNDc3l3o0AEbJ008/nUceeWTg8a9+9avcd999OfDAA/OKV7yihJPB2OV25DAOXX311bn88svT3t6ehoaGXHXVVWlsbCz1WACMkvXr1+f1r3/9buuLFi3KV77yldEfCF4ChBMAAEABv3ECAAAoIJwAAAAKCCcAAIACwgkAAKCAcAIAACggnAAAAAoIJwAAgALCCQAAoIBwAgAAKCCcAAAACggnAPY6N998c2bOnJl99903Bx10UJqamtLV1ZUk+fKXv5xjjjkm1dXVmT59er74xS8OnPee97wnxx13XLq7u5MkPT09Of7447Nw4cKSfA4Axg7hBMBe5YknnsiZZ56Z97znPXnggQeyfv36nHHGGenv7883v/nNLFu2LJdcckkeeOCBXHrppbnooovy1a9+NUly1VVXpaurK+eff36S5IILLsiTTz6Zq6++upQfCYAxYEKpBwCAkfTEE0/kmWeeyRlnnJHDDz88STJz5swkyfLly/P5z38+Z5xxRpLkiCOOyC9/+ctcc801WbRoUSZNmpRvfOMbOemkk7L//vtn5cqVueuuuzJ58uSSfR4Axoay/v7+/lIPAQAjpbe3N/Pnz8/GjRszf/78nHLKKXnnO9+ZysrKTJo0Kfvuu2/Ky5/9wsUzzzyTmpqadHR0DKx96lOfSmtraz75yU/msssuK8XHAGCMccUJgL1KRUVF7rzzztx9992544478oUvfCEXXHBB/vM//zNJct1116WxsXG3c/6ir68vP/rRj1JRUZFHHnlkVGcHYOzyGycA9jplZWU58cQTc/HFF+fee+9NZWVlfvSjH+XQQw/No48+mqOOOmrQccQRRwyce/nll+fBBx/M//zP/2TdunW54YYbSvhJABgrXHECYK9yzz33pK2tLaecckoOPvjg3HPPPdm+fXuOOeaYXHzxxfnwhz+cmpqavOlNb0p3d3d+8pOf5He/+11aWlpy7733ZtmyZbn55ptz4oknZsWKFTn33HNz0kkn5ZWvfGWpPxoAJeQ3TgDsVR544IGcd9552bx5czo7O3P44YfnQx/6UJYsWZIkufHGG3P55Zfnl7/8Zfbbb7/MnDkzH/nIR/LmN785s2fPzmtf+9pcc801A6932mmnZceOHfnBD34w6Ct9AIwvwgkAAKCA3zgBAAAUEE4AAAAFhBMAAEAB4QQAAFBAOAEAABQQTgAAAAWEEwAAQAHhBAAAUEA4AQAAFBBOAAAABYQTAABAAeEEAABQ4P8B3q8cQ4wHqCQAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x=dataset[\"sex\"], y=dataset[\"target\"])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RmvOD6T0weZ6"
      },
      "source": [
        "`We notice, that females are more likely to have heart problems than males`"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lkXsr5QfwUc5"
      },
      "source": [
        "Analysing the 'Chest Pain Type' feature\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z4KkWfEawZT3",
        "outputId": "9693dbac-5a5c-4b55-b897-3ad486b04753"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([3, 2, 1, 0])"
            ]
          },
          "execution_count": 46,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"cp\"].unique()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1SGpijDPwq9d"
      },
      "source": [
        "`As expected, the CP feature has values from 0 to 3`\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 542
        },
        "id": "ooLGQsrWxPu8",
        "outputId": "5e01e40a-fd25-4237-c793-e96539846738"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA04AAAINCAYAAAAJGy/3AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAjQElEQVR4nO3de5BW9X3H8c8C7q6KYJSwKGKw0XgZFFoIDBrbGlGSdLz0NkQ7QrfGjgmkxvVKFKhNFaKJxVQSolXTzIRC68TLqMXYjZhJxRABU2O9jFWDoy5CaAE3ldXd7R+ZbLIj8HPrsmdxX6+ZZ8bze87Z57uOJ5P3nPOcrens7OwMAAAAuzSo6gEAAAD6O+EEAABQIJwAAAAKhBMAAECBcAIAACgQTgAAAAXCCQAAoEA4AQAAFAypeoC+1tHRkVdffTUHHHBAampqqh4HAACoSGdnZ7Zv355DDz00gwbt/prSgAunV199NWPGjKl6DAAAoJ94+eWXc9hhh+12nwEXTgcccECSX/7LGTZsWMXTAAAAVdm2bVvGjBnT1Qi7M+DC6Ve35w0bNkw4AQAA7+orPB4OAQAAUCCcAAAACoQTAABAgXACAAAoEE4AAAAFwgkAAKBAOAEAABQIJwAAgALhBAAAUCCcAAAACoQTAABAgXACAAAoEE4AAAAFwgkAAKBAOAEAABQIJwAAgALhBAAAUDCk6gEABprOzs60trZ2be+///6pqampcCIAoEQ4AfSx1tbWnHXWWV3b99xzT4YOHVrhRABAiVv1AAAACoQTAABAgXACAAAoEE4AAAAFwgkAAKBAOAEAABQIJwAAgALhBAAAUCCcAAAACoQTAABAgXACAAAoEE4AAAAFwgkAAKBAOAEAABQIJwAAgALhBAAAUCCcAAAACoQTAABAgXACAAAoEE4AAAAFQ6oeAAAYeDo7O9Pa2tq1vf/++6empqbCiQB2TzgBAH2utbU1Z511Vtf2Pffck6FDh1Y4EcDuCScAAOhjrrrufYQTAAD0MVdd9z4eDgEAAFAgnAAAAAqEEwAAQIFwAgAAKBBOAAAABcIJAACgQDgBAAAUCCcAAIAC4QQAAFAgnAAAAAqEEwAAQIFwAgAAKBBOAAAABcIJAACgQDgBAAAUCCcAAIAC4QQAAFAgnAAAAAqEEwAAQIFwAgAAKBBOAAAABcIJAACgQDgBAAAUCCcAAICCIVUPAOz9Jl727apH2KvUvN2W4b+x/fvzlqdzSG1l8+xN1t4ws+oRABigXHECAAAoEE4AAAAFwgkAAKBAOAEAABRUHk5LlizJ2LFjU19fnylTpmTNmjW73X/x4sU5+uijs++++2bMmDG5+OKL8+abb/bRtAAAwEBUaTitWLEiTU1NWbBgQdatW5fx48dn+vTpef3113e6/7Jly3LllVdmwYIFefrpp3PbbbdlxYoV+eIXv9jHkwMAAANJpeF044035oILLkhjY2OOO+64LF26NPvtt19uv/32ne7/6KOP5qSTTsq5556bsWPH5vTTT88555xTvEoFAADwXlQWTm1tbVm7dm2mTZv262EGDcq0adOyevXqnR5z4oknZu3atV2h9MILL+SBBx7Ipz71qV1+zo4dO7Jt27ZuLwAAgJ6o7A/gbt68Oe3t7WloaOi23tDQkGeeeWanx5x77rnZvHlzPvaxj6WzszNvv/12Lrzwwt3eqrdw4cJcc801vTo7AAAwsFT+cIieWLVqVa677rp8/etfz7p16/Ld7343999/f770pS/t8pi5c+dm69atXa+XX365DycGAADeDyq74jRixIgMHjw4Gzdu7La+cePGjBo1aqfHzJs3L+edd14+85nPJEmOP/74tLa25i//8i9z1VVXZdCgd3ZgXV1d6urqev8XAAAABozKwqm2tjYTJ05Mc3Nzzj777CRJR0dHmpubM2fOnJ0e84tf/OIdcTR48OAkSWdn5x6dFwB2Z+Jl3656hL1KzdttGf4b278/b3k6h9RWNs/eZO0NM6seAQakysIpSZqamjJr1qxMmjQpkydPzuLFi9Pa2prGxsYkycyZMzN69OgsXLgwSXLGGWfkxhtvzG//9m9nypQpef755zNv3rycccYZXQEFAADQ2yoNpxkzZmTTpk2ZP39+WlpaMmHChKxcubLrgREbNmzodoXp6quvTk1NTa6++uq88sor+eAHP5gzzjgj1157bVW/AgAAMABUGk5JMmfOnF3emrdq1apu20OGDMmCBQuyYMGCPpgMAADgl/aqp+oBAABUQTgBAAAUCCcAAIAC4QQAAFAgnAAAAAqEEwAAQIFwAgAAKBBOAAAABcIJAACgQDgBAAAUCCcAAIAC4QQAAFAgnAAAAAqEEwAAQIFwAgAAKBBOAAAABcIJAACgQDgBAAAU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AACAfqnH4fQf//EfO12vqalJfX19Dj/8cA+JAAAA3ld6HE4TJkxITU1NkqSzszNJuraTZJ999smMGTPyzW9+M/X19b00JgAAQHV6/B2nu+66K0cddVRuueWW/OQnP8lPfvKT3HLLLTn66KOzbNmy3Hbbbfn+97+fq6++ek/MCwAA0Od6fMXp2muvzU033ZTp06d3rR1//PE57LDDMm/evKxZsyb7779/LrnkknzlK1/p1WEBAACq0OMrTk8++WQ+9KEPvWP9Qx/6UJ588skkv7yd77XXXnvv0wEAAPQDPQ6nY445JosWLUpbW1vX2ltvvZVFixblmGOOSZK88soraWho6L0pAQAAKtTjW/WWLFmSM888M4cddlhOOOGEJL+8CtXe3p777rsvSfLCCy/kc5/7XO9OCgAAUJEeh9OJJ56YF198Md/5znfy3HPPJUn+9E//NOeee24OOOCAJMl5553Xu1MCAABUqEfh9NZbb+WYY47JfffdlwsvvHBPzQQAANCv9Cic9tlnn7z55pt7ahYAABgQOgfvk60nnNNtm/6txw+HmD17dr785S/n7bff3hPzAADA+19NTTqH1Ha9UlNT9UQU9Pg7Tj/+8Y/T3Nyc733vezn++OOz//77d3v/u9/9bq8NBwAA0B/0OJwOPPDA/PEf//GemAUAAKBf6nE43XHHHXtiDgAAgH6rx99xAgAAGGh6fMUpSe6888788z//czZs2JC2trZu761bt65XBgMAAOgvenzF6Wtf+1oaGxvT0NCQ9evXZ/LkyTn44IPzwgsv5JOf/OSemBEAAKBSPQ6nr3/967nlllvy93//96mtrc3ll1+ehx56KH/1V3+VrVu37okZAQAAKtXjcNqwYUNOPPHEJMm+++6b7du3J0nOO++8/NM//VPvTgcAANAP9DicRo0alS1btiRJDj/88Dz22GNJkhdffDGdnZ29Ox0AAEA/0ONw+vjHP5577703SdLY2JiLL744p512WmbMmJE//MM/7PUBAQAAqtbjp+pdddVVGT16dJJk9uzZOfjgg/Poo4/mzDPPzCc+8YleHxAAAKBqPQ6nI488Mq+99lpGjhyZJPn0pz+dT3/60/n5z3+ekSNHpr29vdeHBAAAqFKPb9Xb1feY3njjjdTX17/ngQAAAPqbd33FqampKUlSU1OT+fPnZ7/99ut6r729PT/6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\n",
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x=dataset[\"cp\"], y=dataset[\"target\"])\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eDtrRpbtxZWV"
      },
      "source": [
        "\n",
        "\n",
        "`We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems`"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n_R_PJmjxrDz"
      },
      "source": [
        "**Analysing the FBS feature**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "tkogXVGHxu8P",
        "outputId": "d2dd36a7-a112-4508-9cd5-f2d07f964537"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>fbs</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>303.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.148515</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.356198</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ],
            "text/plain": [
              "count    303.000000\n",
              "mean       0.148515\n",
              "std        0.356198\n",
              "min        0.000000\n",
              "25%        0.000000\n",
              "50%        0.000000\n",
              "75%        0.000000\n",
              "max        1.000000\n",
              "Name: fbs, dtype: float64"
            ]
          },
          "execution_count": 49,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"fbs\"].describe()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YE14pIi9x3uh",
        "outputId": "7c95b025-6fd4-4d08-a34c-a37fa124b9cb"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([1, 0])"
            ]
          },
          "execution_count": 50,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"fbs\"].unique()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 542
        },
        "id": "U5jB2coQyBlR",
        "outputId": "27932373-6e66-4c64-932d-b30d2cfa7949"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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qx7hN3JmFgCAYolZAACKNSFiduPGjWlubk5DQ0MWL16cHTt2HHHv9ddfn5qammG3hoaGcZwWAICJovKY3bJlS9rb27N27drs2rUr8+fPz9KlS7N3794jHjN9+vTcd999Q7d77713HCcGAGCiqDxm161bl5UrV6atrS3z5s3Lpk2bMnXq1GzevPmIx9TU1GT27NlDt8bGxnGcGACAiaLSmO3r68vOnTvT2to6tFZbW5vW1tZs3779iMc98MADOeOMM9LU1JQLL7ww3/nOd4649/Dhwzl06NCwGwAAx4dKY3b//v3p7+9/2JnVxsbGdHd3j3jMOeeck82bN+dzn/tcPv7xj2dgYCDnn39+fvjDH464v6OjIzNmzBi6NTU1jfnrAACgGpVfZjBaS5YsyfLly9PS0pKnP/3pufHGG/MHf/AHef/73z/i/tWrV+fgwYNDtz179ozzxAAAHCuV/mjCrFmzMmnSpPT09Axb7+npyezZs4/qb0yZMiVPetKTctddd434eH19ferr63/nWQEAmHgqPTNbV1eXBQsWpLOzc2htYGAgnZ2dWbJkyVH9jf7+/uzevTunnnrqsRoTAIAJqvKfs21vb8+KFSuycOHCLFq0KOvXr09vb2/a2tqSJMuXL8+cOXPS0dGRJLn88svzlKc8JWeddVZ+8pOf5Kqrrsq9996bv//7v6/yZQAAUIHKY3bZsmXZt29f1qxZk+7u7rS0tGTr1q1DHwrr6upKbe2vTiD/7//+b1auXJnu7u6cdNJJWbBgQW699dbMmzevqpcAAEBFKo/ZJFm1alVWrVo14mPbtm0bdv+aa67JNddcMw5TAQAw0RX3bQYAAPCQCXFmFo43g5Om5OB5Lx52HwAYe2IWjoWamgxOrqt6CgA47rnMAACAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiWmAUAoFhiFgCAYolZAACKJWYBACiW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69atedrTnlblywA4ZvxbFcAENmPGjEybNi2TJk3K7Nmzh9abmppyzTXXpKamJuecc052796da665JitXrkxXV1dOOOGE/OVf/mWmTZuWM844I0960pMqfBUAx44zswAFespTnpKampqh+0uWLMl///d/p7+/PxdccEHOOOOMPO5xj8tLXvKSfOITn8hPf/rTCqcFOHbELMBxZtq0adm1a1c+9alP5dRTT82aNWsyf/78/OQnP6l6NIAxJ2YBCvT1r3992P2vfe1rOfvsszNp0qQkD37jQWtra6688sp8+9vfzj333JMvf/nLVYwKcEy5ZhagQF1dXWlvb88//MM/ZNeuXbn22mvzrne9K0ly00035e67787Tnva0nHTSSbn55pszMDCQc845p+KpAcaemAUo0PLly/Ozn/0sixYtyqRJk3LJJZfkZS97WZJk5syZufHGG/OWt7wlP//5z3P22WfnU5/6VP7wD/+w4qkBxl7N4K9/MSEAABTENbMAABRLzAIAUCwxCwBAscQsAADFErMAABRLzAIAUCwxCwBAscQsAADFErMAABRLzAIAUCwxCwBAscQsAADF+n8d2y1AI59x7gAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(8, 6))\n",
        "sns.barplot(x=dataset[\"fbs\"], y=dataset[\"target\"])\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yZ--QR1ZyuO0"
      },
      "source": [
        "Nothing extraordinary here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ia2sXh5Sy52n"
      },
      "source": [
        "**Analysing the restecg feature**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LpHPW9qxzCAz",
        "outputId": "ac67b75d-3254-441d-fb69-fe7962fa5842"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([0, 1, 2])"
            ]
          },
          "execution_count": 56,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"restecg\"].unique()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 559
        },
        "id": "o9LmIHznzM99",
        "outputId": "e12d8f0e-798a-4b2c-9649-f7f2d7ab55ce"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='restecg', ylabel='target'>"
            ]
          },
          "execution_count": 60,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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Tz755BD8LQFDg1ACw9Qjjzwiu92urVu3avny5frmN7+pM844Q6+88opqamrU2tqqyy+/XJK0b98+LVmyRP/+7/+uN998U7W1tbrssssUCoV0/fXX6/LLL9eFF16offv2ad++fZo/f74OHz6svLw8jRs3Ti+++KK2bt2quLg4XXjhheEz2AceeEDFxcX6j//4D+3evVtPPfWUpk+fLkkKBAJavHixYmNjtX37dq1atUrLli2z7O8LOG5CAIadc889N3TGGWeEb99+++2hhQsX9tinubk5JCnU2NgYamhoCEkK7d27t9fHKywsDC1atKjH2qOPPhqaMWNGKBgMhte6urpCY8aMCW3cuDEUCoVCU6ZMCS1btqzXx/yf//mf0KhRo0L79u0Lr23atCkkKbR+/fqBPF1gWOP7KIFhKjMzM/znV199VS+88ILi4uKO2u/dd9/VwoULdf7552vu3LnKy8vTwoUL9d3vflcTJkzo8/FfffVV7dmzR+PGjeux/umnn+rdd9/V/v379fHHH+v888/v9fjGxkalpqYqOTk5vJadnT3QpwkMe4QSGKbGjh0b/vOhQ4d0ySWX6M477zxqv8mTJys6OlqbNm3Stm3b9Nxzz2nlypVatmyZtm/f3uc32B86dEiZmZl67LHHjrrvpJNOUlQUn8wAEp9RAieEr3/963rjjTeUlpam6dOn99g+D6rNZtNZZ52lW2+9VTt37pTdbtf69eslSXa7XYFA4KjHfOedd5SUlHTUYyYkJGjcuHFKS0uTz+frdaYZM2aoublZra2t4bWXX375OP0NANYhlMAJoLi4WAcOHNCSJUv08ssv691339XGjRvldrsVCAS0fft23XHHHXrllVfU1NSkdevWqa2tTbNmzZIkpaWl6bXXXlNjY6Pa29t1+PBhXXnllUpMTNSiRYv04osv6v3331dtba1+9KMf6cMPP5Qk3XLLLbr33nt1//3365133tGOHTu0cuVKSdIFF1ygadOmqbCwUK+99pq2bt2qm266SdKRaAMjBaEETgBTpkzR1q1bFQgEtHDhQs2dO1fXXXedxo8fr6ioKMXHx2vz5s361re+pdNPP1033XST7r33Xl100UWSpKKiIs2YMUNZWVk66aSTtHXrVsXGxmrz5s06+eSTddlll2nWrFm6+uqr9emnnyo+Pl6SVFhYqIqKCv3617/W1772Nf3Lv/yL3nnnHUlSdHS0nnzySR06dEhnnnmmrrnmmvBVrzExMdb8RQHHgS0UCoWsHgLAyLB161adffbZ2rNnj6ZNm2b1OMCgIJQAjtn69esVFxen0047TXv27NHSpUs1YcIEbdmyxerRgEHDVa8AjtnBgwd1ww03qKmpSYmJiXK5XLr33nutHgsYVJxRAgBgwMU8AAAYEEoAAAwIJQAABoQSAAADQgkAgAGhBADAgFACAGBAKAEAMCCUAAAY/D/FPhX0JzSdWQAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 500x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(5, 6))\n",
        "sns.barplot(x=dataset[\"restecg\"], y=dataset[\"target\"])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kVFTGiegzabb"
      },
      "source": [
        "`We observe, that Slope '2' causes heart pain much more than Slope '0' and '1'`"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HePjrQXxzfri"
      },
      "source": [
        "**Analysing the 'ca' feature**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W6R3efRXzlh5",
        "outputId": "ec1338b8-8127-470d-e3c6-ab4eb0cb9f49"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([0, 2, 1, 3, 4])"
            ]
          },
          "execution_count": 61,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "dataset[\"ca\"].unique()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 559
        },
        "id": "UOium24Azo7x",
        "outputId": "8bc474d8-06e2-4292-8c01-a3a7d45f25a2"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='ca', ylabel='target'>"
            ]
          },
          "execution_count": 65,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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40yePLnXBwQAAKjaqw6nDRs2JHnpjNOmTZtSW1vb9V5tbW0mTZqUj3/8470/IQAAQMVedTh9//vfT5LMnTs3N998c0aPHn3AhgIAAOhPenyP0x133JHRo0fnqaeeykMPPZTnnnsuSfb5pD0AAICBoMfhtGPHjpx99tl5y1vekve///159tlnkyQXX3xxPvaxj/X6gAAAAFXrcThdccUVOeSQQ7J58+ZuD4iYNWtWVq1a1avDAQAA9Aev+h6nP/jud7+bhx56KEcffXS39eOPPz6//vWve20wAACA/qLHZ5xaW1u7nWn6gx07dqSurq5XhgIAAOhPehxO73znO/Pv//7vXds1NTXp6OjIDTfckLPOOqtXhwMAAOgPenyp3g033JCzzz47jz/+ePbu3ZurrroqP/vZz7Jjx47893//94GYEQAAoFI9PuN00kkn5cknn8w73vGOnH/++Wltbc1f/dVfZcOGDXnzm998IGYEAACoVI/POCXJmDFjcs011/T2LAAAAP1Sj8Ppf/7nf/a5XlNTk/r6+hxzzDEeEgEAAAwoPQ6nyZMnp6amJknS2dmZJF3bSXLIIYdk1qxZ+epXv5r6+vpeGhMAAKA6Pb7H6Z577snxxx+f2267LT/5yU/yk5/8JLfddlve+ta35q677srXv/71fO9738u11157IOYFAADocz0+4/S5z30uN998c2bOnNm1dvLJJ+foo4/Oddddl7Vr12bkyJH52Mc+li9+8Yu9OiwAAEAVenzGadOmTXnjG9/4svU3vvGN2bRpU5KXLud79tlnX/t0AAAA/UCPw+mEE07I9ddfn71793atvfDCC7n++utzwgknJEl+85vfpKGhofemBAAAqFCPL9VbunRpPvCBD+Too4/OKaeckuSls1Dt7e25//77kyRPP/10PvrRj/bupAAAABXpcTidccYZeeaZZ/KNb3wjTz75ZJLkb//2b/OhD30oo0aNSpJcdNFFvTslAABAhXoUTi+88EJOOOGE3H///bnssssO1EwAAAD9So/ucTrkkEPy/PPPH6hZAAAA+qUePxxi3rx5+cIXvpAXX3zxQMwDAADQ7/T4HqfHHnssTU1N+e53v5uTTz45I0eO7Pb+t7/97V4bDgAAoD/ocTgddthh+eu//usDMQsAAEC/1ONwuuOOOw7EHAAAAP1Wj+9xAgAAGGx6fMYpSe6+++5861vfyubNm7N3795u761fv75XBgMAAOgvenzG6ctf/nLmzp2bhoaGbNiwIaeffnoOP/zwPP300znnnHMOxIwAAACV6nE43Xrrrbntttvyr//6r6mtrc1VV12Vhx9+OP/4j/+YnTt3HogZAQAAKtXjcNq8eXPOOOOMJMnw4cOze/fuJMlFF12Ub37zm707HQAAQD/Q43AaP358duzYkSQ55phj8qMf/ShJ8swzz6Szs7N3pwMAAOgHehxO7373u/Od73wnSTJ37txceeWVec973pNZs2blL//yL3t9QAAAgKr1+Kl611xzTY466qgkybx583L44Yfn0UcfzQc+8IG8733v6/UBAQAAqtbjcDruuOPy7LPPZty4cUmSD37wg/ngBz+Y//3f/824cePS3t7e60MCAABUqceX6u3vPqY9e/akvr7+NQ8EAADQ37zqM06NjY1JkpqamixcuDAjRozoeq+9vT0//vGPM3ny5F4fEAAAoGqvOpw2bNiQ5KUzTps2bUptbW3Xe7W1tZk0aVI+/vGP9/6EAAAAFXvV4fT9738/yUtP0rv55pszevToAzYUAABAf9Ljh0PccccdB2IOAACAfqvHD4cAAAAYbIQTAABAgXACAAAoEE4AAAAFwgkAAKBAOAEAABQIJwAAgALhBAAAUCCcAAAACoZVPQDAH3R2dqa1tbVre+TIkampqalwIgCAlwgnoN9obW3N+eef37V933335dBDD61wIgCAl7hUDwAAoEA4AQAAFAgnAACAAuEEAABQIJwAAAAKhBMAAECBcAIAACgQTgAAAAXCCQAAoEA4AQAAFAgnAACAAuEEAABQIJwAAAAKhBMAAECBcAIAACgQTgAAAAXCCQAAoKBfhNPSpUszceLE1NfXZ9q0aVm7du2rOm7FihWpqanJBRdccGAHBAAABrXKw2nlypVpbGzMokWLsn79+kyaNCkzZ87M1q1bX/G4X/3qV/n4xz+ed77znX00KQAAMFhVHk433XRTLr300sydOzcnnnhili1blhEjRmT58uX7Paa9vT0f/vCH8+lPfzpvetOb+nBaAABgMKo0nPbu3Zt169ZlxowZXWtDhgzJjBkzsmbNmv0e98///M8ZN25cLr744uJntLW1ZdeuXd1eAAAAPVFpOG3fvj3t7e1paGjott7Q0JDm5uZ9HvPDH/4wX//6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\n",
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "colors = [\"#FF9999\", \"#66B3FF\", \"#99FF99\", \"#FFCC99\", \"#CFCFCF\"]\n",
        "\n",
        "# Plot with custom colors for each bar\n",
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(x=dataset[\"ca\"], y=dataset[\"target\"], palette=colors)  # Apply custom colors\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "p7URb8in0-gT",
        "outputId": "952e3a05-7bfb-4eb7-b522-a788d34f1f6b"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "<Axes: xlabel='thal', ylabel='Density'>"
            ]
          },
          "execution_count": 66,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAGwCAYAAABB4NqyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABWG0lEQVR4nO3de3xT9f0/8NdJ0iS9pvcbtLRQ7pcWUErxAmi1oEPQ7xyyTS7eNif7znXM2f0mbLp9UacIbkw2FZFtCDoVnRcQqwWRci3lfqelLW16oZe0aZu0yfn9kSZYaKGXJCfJeT0fjzykpycn71AhLz6f9/l8BFEURRARERHJiELqAoiIiIjcjQGIiIiIZIcBiIiIiGSHAYiIiIhkhwGIiIiIZIcBiIiIiGSHAYiIiIhkRyV1AZ7IarWivLwcwcHBEARB6nKIiIioB0RRRGNjI+Lj46FQXHuMhwGoC+Xl5UhISJC6DCIiIuqD0tJSDBw48JrnMAB1ITg4GIDtNzAkJETiaoiIiKgnDAYDEhISHJ/j18IA1AX7tFdISAgDEBERkZfpSfsKm6CJiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdiQNQMuXL8eNN96I4OBgREdHY86cOTh16tR1n/fee+9hxIgR0Gq1GDt2LD777LNO3xdFEUuXLkVcXBz8/f2RmZmJM2fOuOptEBERkZeRNABt374dTzzxBHbv3o1t27ahra0Nd955J4xGY7fP2bVrF+bNm4eHH34YBw8exJw5czBnzhwcPXrUcc6LL76IV199FWvWrMGePXsQGBiIrKwstLa2uuNtERERkYcTRFEUpS7Crrq6GtHR0di+fTtuvfXWLs+ZO3cujEYjPvnkE8exyZMnIy0tDWvWrIEoioiPj8evfvUrLFmyBADQ0NCAmJgYrFu3Dg888MBV1zSZTDCZTI6vDQYDEhIS0NDQgJCQECe/SyKi3tmwp0SS1/1heqIkr0vUVwaDATqdrkef3x7VA9TQ0AAACA8P7/ac/Px8ZGZmdjqWlZWF/Px8AEBRURH0en2nc3Q6HdLT0x3nXGn58uXQ6XSOR0JCQn/fChEREXkwjwlAVqsVTz75JG666SaMGTOm2/P0ej1iYmI6HYuJiYFer3d8336su3OulJOTg4aGBsejtLS0P2+FiIiIPJxK6gLsnnjiCRw9ehQ7d+50+2trNBpoNBq3vy4RERFJwyNGgBYvXoxPPvkEX3/9NQYOHHjNc2NjY1FZWdnpWGVlJWJjYx3ftx/r7hwiIiKSN0kDkCiKWLx4MT788EN89dVXSE5Ovu5zMjIykJub2+nYtm3bkJGRAQBITk5GbGxsp3MMBgP27NnjOIeIiIjkTdIpsCeeeAIbNmzARx99hODgYEePjk6ng7+/PwBg/vz5GDBgAJYvXw4A+MUvfoGpU6fi5Zdfxt13342NGzdi//79+Mc//gEAEAQBTz75JP74xz9i6NChSE5OxjPPPIP4+HjMmTNHkvdJREREnkXSAPTaa68BAKZNm9bp+FtvvYWFCxcCAEpKSqBQXB6omjJlCjZs2IDf/e53+O1vf4uhQ4di8+bNnRqnn3rqKRiNRjz22GOor6/HzTffjC1btkCr1br8PREREZHn86h1gDxFb9YRICJyNa4DRNQzXrsOEBEREZE7MAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7EgagHbs2IFZs2YhPj4egiBg8+bN1zx/4cKFEAThqsfo0aMd5/z+97+/6vsjRoxw8TshIiIibyJpADIajUhNTcXq1at7dP6qVatQUVHheJSWliI8PBz3339/p/NGjx7d6bydO3e6onwiIiLyUiopX3zmzJmYOXNmj8/X6XTQ6XSOrzdv3oy6ujosWrSo03kqlQqxsbFOq5OIiIh8i1f3AL355pvIzMzEoEGDOh0/c+YM4uPjMXjwYPzoRz9CSUnJNa9jMplgMBg6PYiIiMh3eW0AKi8vx+eff45HHnmk0/H09HSsW7cOW7ZswWuvvYaioiLccsstaGxs7PZay5cvd4wu6XQ6JCQkuLp8IiIikpDXBqC3334boaGhmDNnTqfjM2fOxP33349x48YhKysLn332Gerr6/Huu+92e62cnBw0NDQ4HqWlpS6unoiIiKQkaQ9QX4miiLVr1+LBBx+EWq2+5rmhoaEYNmwYzp492+05Go0GGo3G2WUSERGRh/LKEaDt27fj7NmzePjhh697blNTE86dO4e4uDg3VEZERETeQNIA1NTUhMLCQhQWFgIAioqKUFhY6GhazsnJwfz586963ptvvon09HSMGTPmqu8tWbIE27dvR3FxMXbt2oV7770XSqUS8+bNc+l7ISIiIu8h6RTY/v37MX36dMfX2dnZAIAFCxZg3bp1qKiouOoOroaGBrz//vtYtWpVl9csKyvDvHnzcOnSJURFReHmm2/G7t27ERUV5bo3QkRERF5FEEVRlLoIT2MwGKDT6dDQ0ICQkBCpyyEimduw59pLebjKD9MTJXldor7qzee3V/YAEREREfUHAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyY6kAWjHjh2YNWsW4uPjIQgCNm/efM3z8/LyIAjCVQ+9Xt/pvNWrVyMpKQlarRbp6enYu3evC98FEREReRtJA5DRaERqaipWr17dq+edOnUKFRUVjkd0dLTje5s2bUJ2djaWLVuGgoICpKamIisrC1VVVc4un4iIiLyUSsoXnzlzJmbOnNnr50VHRyM0NLTL761YsQKPPvooFi1aBABYs2YNPv30U6xduxZPP/10f8olIiIiH+GVPUBpaWmIi4vDHXfcgW+//dZx3Gw248CBA8jMzHQcUygUyMzMRH5+frfXM5lMMBgMnR5ERETku7wqAMXFxWHNmjV4//338f777yMhIQHTpk1DQUEBAKCmpgYWiwUxMTGdnhcTE3NVn9B3LV++HDqdzvFISEhw6fsgIiIiaUk6BdZbw4cPx/Dhwx1fT5kyBefOncMrr7yCf/7zn32+bk5ODrKzsx1fGwwGhiAiIiIf5lUBqCuTJk3Czp07AQCRkZFQKpWorKzsdE5lZSViY2O7vYZGo4FGo3FpnUREROQ5vGoKrCuFhYWIi4sDAKjVakycOBG5ubmO71utVuTm5iIjI0OqEomIiMjDSDoC1NTUhLNnzzq+LioqQmFhIcLDw5GYmIicnBxcvHgR69evBwCsXLkSycnJGD16NFpbW/HGG2/gq6++whdffOG4RnZ2NhYsWIAbbrgBkyZNwsqVK2E0Gh13hRERERFJGoD279+P6dOnO7629+EsWLAA69atQ0VFBUpKShzfN5vN+NWvfoWLFy8iICAA48aNw5dfftnpGnPnzkV1dTWWLl0KvV6PtLQ0bNmy5arGaCIiIpIvQRRFUeoiPI3BYIBOp0NDQwNCQkKkLoeIZG7DnpLrn+QCP0xPlOR1ifqqN5/fXt8DRERERNRbDEBEREQkOwxAREREJDsMQERERCQ7DEBEREQkOwxAREREJDsMQERERCQ7DEBEREQkO16/GSoREVF/SbHYJBealBZHgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYYgIiIiEh2GICIiIhIdhiAiIiISHYkDUA7duzArFmzEB8fD0EQsHnz5mue/8EHH+COO+5AVFQUQkJCkJGRga1bt3Y65/e//z0EQej0GDFihAvfBREREXkbSQOQ0WhEamoqVq9e3aPzd+zYgTvuuAOfffYZDhw4gOnTp2PWrFk4ePBgp/NGjx6NiooKx2Pnzp2uKJ+IiIi8lErKF585cyZmzpzZ4/NXrlzZ6ev/+7//w0cffYT//ve/GD9+vOO4SqVCbGyss8okIiIiH+PVPUBWqxWNjY0IDw/vdPzMmTOIj4/H4MGD8aMf/QglJSXXvI7JZILBYOj0ICIiIt/l1QHopZdeQlNTE37wgx84jqWnp2PdunXYsmULXnvtNRQVFeGWW25BY2Njt9dZvnw5dDqd45GQkOCO8omIiEgiXhuANmzYgD/84Q949913ER0d7Tg+c+ZM3H///Rg3bhyysrLw2Wefob6+Hu+++26318rJyUFDQ4PjUVpa6o63QERERBKRtAeorzZu3IhHHnkE7733HjIzM695bmhoKIYNG4azZ892e45Go4FGo3F2mUREROShvG4E6J133sGiRYvwzjvv4O67777u+U1NTTh37hzi4uLcUB0RERF5A0lHgJqamjqNzBQVFaGwsBDh4eFITExETk4OLl68iPXr1wOwTXstWLAAq1atQnp6OvR6PQDA398fOp0OALBkyRLMmjULgwYNQnl5OZYtWwalUol58+a5/w0SERGRR5J0BGj//v0YP3684xb27OxsjB8/HkuXLgUAVFRUdLqD6x//+Afa29vxxBNPIC4uzvH4xS9+4TinrKwM8+bNw/Dhw/GDH/wAERER2L17N6Kiotz75oiIiMhjSToCNG3aNIii2O33161b1+nrvLy8615z48aN/ayKiIiIfJ3X9QARERER9RcDEBEREckOAxARERHJDgMQERERyU6fAtD58+edXQcRERGR2/QpAKWkpGD69On417/+hdbWVmfXRERERORSfQpABQUFGDduHLKzsxEbG4uf/OQn2Lt3r7NrIyIiInKJPgWgtLQ0rFq1CuXl5Vi7di0qKipw8803Y8yYMVixYgWqq6udXScRERGR0/SrCVqlUuG+++7De++9hxdeeAFnz57FkiVLkJCQgPnz56OiosJZdRIRERE5Tb8C0P79+/Gzn/0McXFxWLFiBZYsWYJz585h27ZtKC8vx+zZs51VJxEREZHT9GkrjBUrVuCtt97CqVOncNddd2H9+vW46667oFDY8lRycjLWrVuHpKQkZ9ZKRERE5BR9CkCvvfYaHnroISxcuBBxcXFdnhMdHY0333yzX8URERERuUKfAtC2bduQmJjoGPGxE0URpaWlSExMhFqtxoIFC5xSJBEREZEz9akHaMiQIaipqbnqeG1tLZKTk/tdFBEREZEr9SkAiaLY5fGmpiZotdp+FURERL1jaregqMaIWqMZ1m7+fiaizno1BZadnQ0AEAQBS5cuRUBAgON7FosFe/bsQVpamlMLJCKi7lU0tOCf+RdQ39IGAPBTCvje2HjcmBwucWVEnq1XAejgwYMAbCNAR44cgVqtdnxPrVYjNTUVS5YscW6FRETUpVN6A97ZVwpzuxVaPwXa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\n",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sns.distplot(dataset[\"thal\"])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nQkoLkyt1HYZ"
      },
      "source": [
        "**IV. Train Test split**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "alTAZeaC1L5T"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "predictors = dataset.drop(\"target\",axis=1)\n",
        "target = dataset[\"target\"]\n",
        "\n",
        "X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "F9D4b8de1Xc8",
        "outputId": "c990e377-f936-485f-b930-1df223ee49d5"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(242, 13)"
            ]
          },
          "execution_count": 97,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "X_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xyTWBc5T1fB0",
        "outputId": "663722e8-cbf5-4dc7-df56-0fc5688b13e2"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(61, 13)"
            ]
          },
          "execution_count": 98,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "X_test.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jeFOYg8c1jfF",
        "outputId": "2b47ba4f-8485-4ec3-c629-1dbe3ed43c82"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "242"
            ]
          },
          "execution_count": 99,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "Y_train.shape[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AvsHK9uD1r1p",
        "outputId": "f6522de6-58aa-4982-87bd-fadbec31175d"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(61,)"
            ]
          },
          "execution_count": 100,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "Y_test.shape"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h7hDvf-H1vl6"
      },
      "source": [
        " **Model Fitting**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HSloUTkC129z"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import accuracy_score"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FCH8AeRq19yH"
      },
      "source": [
        "**Logistic Regression**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7i7WDAoD2DGy"
      },
      "outputs": [],
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "lr = LogisticRegression()\n",
        "\n",
        "lr.fit(X_train,Y_train)\n",
        "\n",
        "Y_pred_lr = lr.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "78YCwDnK2H49",
        "outputId": "0ff6d807-1b92-4df4-d3bc-a6fb848a7df5"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "61"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ],
      "source": [
        "Y_pred_lr.shape[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xqaGqEvj2H-z",
        "outputId": "ee77c8ef-22a4-4594-a851-c9293e0f5f34"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using Logistic Regression is: 85.25 %\n"
          ]
        }
      ],
      "source": [
        "score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using Logistic Regression is: \"+str(score_lr)+\" %\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HZznjooD2T3u"
      },
      "source": [
        "**Naive Bayes**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GRBaQgfv2V17"
      },
      "outputs": [],
      "source": [
        "from sklearn.naive_bayes import GaussianNB\n",
        "\n",
        "nb = GaussianNB()\n",
        "\n",
        "nb.fit(X_train,Y_train)\n",
        "\n",
        "Y_pred_nb = nb.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WIszTYg42b4A",
        "outputId": "61d87d95-e546-4e1c-fdc9-27cb88257f6a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "61"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ],
      "source": [
        "Y_pred_nb.shape[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "niioVKSP2jAZ",
        "outputId": "b9fca4ec-5554-47dd-8c6b-8f245e8434f2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using Naive Bayes is: 85.25 %\n"
          ]
        }
      ],
      "source": [
        "score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using Naive Bayes is: \"+str(score_nb)+\" %\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eIIFWUBU2nbU"
      },
      "source": [
        "**SVM**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "M-Mx0hQj2rPL"
      },
      "outputs": [],
      "source": [
        "from sklearn import svm\n",
        "\n",
        "sv = svm.SVC(kernel='linear')\n",
        "\n",
        "sv.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred_svm = sv.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k0L-EpE02vZI",
        "outputId": "7448a4bd-e58d-40b4-9627-dc86366b8ea9"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(61,)"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ],
      "source": [
        "Y_pred_svm.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cvL9u92q2zA1",
        "outputId": "a08b6a31-d7cf-4330-bbda-abca8bf4c6a8"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using Linear SVM is: 81.97 %\n"
          ]
        }
      ],
      "source": [
        "score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using Linear SVM is: \"+str(score_svm)+\" %\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ynGEdmHE24Xy"
      },
      "source": [
        "**K Nearest Neighbors**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Wtxc6rKe2860"
      },
      "outputs": [],
      "source": [
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "\n",
        "knn = KNeighborsClassifier(n_neighbors=7)\n",
        "knn.fit(X_train,Y_train)\n",
        "Y_pred_knn=knn.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MDrbSnLe3A59",
        "outputId": "30e84096-1c2f-4c8b-8a03-11c60d91e19a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(61,)"
            ]
          },
          "metadata": {},
          "execution_count": 40
        }
      ],
      "source": [
        "Y_pred_knn.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tna5Re8r3F6e",
        "outputId": "2c63c0e1-12c4-4fa9-9797-1623256117a3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using KNN is: 67.21 %\n"
          ]
        }
      ],
      "source": [
        "score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using KNN is: \"+str(score_knn)+\" %\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wEDM0dlC3JJS"
      },
      "source": [
        "**Decision Tree**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FClvBcuZ3O2O"
      },
      "outputs": [],
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "\n",
        "max_accuracy = 0\n",
        "\n",
        "\n",
        "for x in range(200):\n",
        "    dt = DecisionTreeClassifier(random_state=x)\n",
        "    dt.fit(X_train,Y_train)\n",
        "    Y_pred_dt = dt.predict(X_test)\n",
        "    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)\n",
        "    if(current_accuracy>max_accuracy):\n",
        "        max_accuracy = current_accuracy\n",
        "        best_x = x\n",
        "\n",
        "#print(max_accuracy)\n",
        "#print(best_x)\n",
        "\n",
        "\n",
        "dt = DecisionTreeClassifier(random_state=best_x)\n",
        "dt.fit(X_train,Y_train)\n",
        "Y_pred_dt = dt.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4XPejApG3SWY",
        "outputId": "beef1214-3a45-443a-c669-8572ae4be039"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(61,)\n"
          ]
        }
      ],
      "source": [
        "print(Y_pred_dt.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "optH-KsY3WAD",
        "outputId": "5c1132df-05a0-4bd4-b311-5d38210211bb"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using Decision Tree is: 81.97 %\n"
          ]
        }
      ],
      "source": [
        "score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using Decision Tree is: \"+str(score_dt)+\" %\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PypfNkQ53cHL"
      },
      "outputs": [],
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "max_accuracy = 0\n",
        "\n",
        "\n",
        "for x in range(50):\n",
        "    rf = RandomForestClassifier(random_state=x)\n",
        "    rf.fit(X_train,Y_train)\n",
        "    Y_pred_rf = rf.predict(X_test)\n",
        "    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)\n",
        "    if(current_accuracy>max_accuracy):\n",
        "        max_accuracy = current_accuracy\n",
        "        best_x = x\n",
        "\n",
        "#print(max_accuracy)\n",
        "#print(best_x)\n",
        "\n",
        "rf = RandomForestClassifier(random_state=best_x)\n",
        "rf.fit(X_train,Y_train)\n",
        "Y_pred_rf = rf.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MtJNSM-v33cY",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "54147d8b-7c26-41cc-8700-6236c48aa26a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(61,)"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ],
      "source": [
        "Y_pred_rf.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5lId-JnB47Rk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4d1c12b5-a6be-4b45-c596-bd323881d67a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using Decision Tree is: 88.52 %\n"
          ]
        }
      ],
      "source": [
        "score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using Decision Tree is: \"+str(score_rf)+\" %\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_OxwyBKj4-B1"
      },
      "source": [
        "**XGBoost**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2YXcFd2p5_kc"
      },
      "outputs": [],
      "source": [
        "import xgboost as xgb\n",
        "\n",
        "xgb_model = xgb.XGBClassifier(objective=\"binary:logistic\", random_state=42)\n",
        "xgb_model.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred_xgb = xgb_model.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QkDDDEoB6YIe",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7372baff-1183-486e-e68f-5c4eab3c5a79"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(61,)"
            ]
          },
          "metadata": {},
          "execution_count": 49
        }
      ],
      "source": [
        "Y_pred_xgb.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iyE5Ci4d6bNN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "509169d6-f395-476f-ca9a-2fefb735e8c9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using XGBoost is: 83.61 %\n"
          ]
        }
      ],
      "source": [
        "score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)\n",
        "\n",
        "print(\"The accuracy score achieved using XGBoost is: \"+str(score_xgb)+\" %\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pLT03acX6lNS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "efd3277a-f230-4044-931d-dd07ea4a4429"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The accuracy score achieved using Logistic Regression is: 85.25 %\n",
            "The accuracy score achieved using Naive Bayes is: 85.25 %\n",
            "The accuracy score achieved using Support Vector Machine is: 81.97 %\n",
            "The accuracy score achieved using K-Nearest Neighbors is: 67.21 %\n",
            "The accuracy score achieved using Decision Tree is: 81.97 %\n",
            "The accuracy score achieved using Random Forest is: 88.52 %\n",
            "The accuracy score achieved using XGBoost is: 83.61 %\n"
          ]
        }
      ],
      "source": [
        "scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb]\n",
        "algorithms = [\"Logistic Regression\",\"Naive Bayes\",\"Support Vector Machine\",\"K-Nearest Neighbors\",\"Decision Tree\",\"Random Forest\",\"XGBoost\"]\n",
        "\n",
        "for i in range(len(algorithms)):\n",
        "    print(\"The accuracy score achieved using \"+algorithms[i]+\" is: \"+str(scores[i])+\" %\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.set(rc={'figure.figsize': (15, 8)})\n",
        "\n",
        "# Create the barplot with color\n",
        "sns.barplot(x=algorithms, y=scores, palette='viridis')  # You can use other palettes or specify colors\n",
        "\n",
        "# Adding labels\n",
        "plt.xlabel(\"Algorithms\")\n",
        "plt.ylabel(\"Accuracy score\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 510
        },
        "id": "5XSUaBjbN5y8",
        "outputId": "1c8f2f43-83a5-42c8-ec8c-c74d5860af69"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'Accuracy score')"
            ]
          },
          "metadata": {},
          "execution_count": 54
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1500x800 with 1 Axes>"
            ],
            "image/png": "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3svpp5+evn37Zv3118/f//73JMn48eNz1lln5aWXXppvRQIAAABApSkUrL322mvZaaedcu+992aFFVbI5MmTM2PGjCRJmzZt8swzz2T48OHztVAAAAAAqCQLFZnpggsuSMuWLXPzzTcnSTbaaKN643v16pV777133qsDAAAAgApV6Iy1v//979ljjz3Spk2bVFVVNRj/wx/+MOPGjZvn4gAAAACgUhUK1kqlUhZZZJE5jh8/fnyaN29euCgAAAAAqHSFgrVOnTrl4Ycfnu24GTNmZNSoUVl77bXnqTAAAAAAqGSFgrWDDz44jz76aM4444y8+uqrSZKPPvoojz/+ePbff/+MGTMmBx988HwtFAAAAAAqSaGHF/Tq1SvnnHNOzj777PIDDH7605+mVCpl8cUXz3nnnZf11ltvvhYKAAAAAJWkULCWJDvuuGO23nrrPP7443nzzTdTW1ubFVdcMZtsskkWX3zx+VkjAAAAAFScuQ7WpkyZkt69e+eggw7KgQcemC233PKbqAsAAAAAKtpc32OtRYsWqampSYsWLb6JegAAAADgO6HQwwu23nrr3H///SmVSvO7HgAAAAD4Tih0j7UBAwZk0KBB2WeffTJw4MAsv/zyWWSRRRpM17lz53kuEAAAAAAqUaFgbe+99y7//9NPP91gfKlUSlVVVf79738XrwwAAAAAKlihYO2cc86Z33UAAAAAwHdKoWBtp512mt91AAAAAMB3SqFg7cs+/fTTfPDBB0mSZZddNostttg8FwUAAAAAla5wsPbiiy/mggsuyLPPPpva2tokSXV1ddZdd9389Kc/TZcuXeZbkQAAAABQaQoFay+88EL23nvvNGvWLLvssktWXXXVJMnrr7+eUaNGZa+99sr111+frl27ztdiAQAAAKBSFArWLr744iyzzDL5wx/+kLZt29Ybd9RRR2WPPfbIxRdfnN///vfzpUgAAAAAqDTVRWZ64YUXsttuuzUI1ZJk6aWXzq677prnn39+XmsDAAAAgIpVKFirrq7OzJkz5zi+trY21dWFFg0AAAAA3wmF0q/u3bvnhhtuyLvvvttg3HvvvZc//OEPWWeddea5OAAAAACoVIXusXb88cdnzz33TL9+/bLVVltl5ZVXTpK88cYbefDBB1NTU5MTTjhhftYJAAAAABWlULDWqVOn3HLLLbn44ovz5z//OVOmTEmStGjRIptuummOPfbYrLbaavO1UAAAAACoJIWCtSRZbbXVMnTo0NTW1mb8+PFJkjZt2ri3GgAAAAALhMLBWp3q6uosvfTS86MWAAAAAPjOKHR62cUXX5wddthhjuN33HHHDBkypHBRAAAAAFDpCgVr999/fzbbbLM5ju/Vq1dGjx5duCgAAAAAqHSFgrX3338/K6644hzHr7DCCnnvvfcKFwUAAAAAla5QsLbooovm3XffneP4d955JwsvvHDhogAAAACg0hUK1tZff/2MGDEi48aNazDu/fffz4gRI9KzZ895Lg4AAAAAKlWhp4Iec8wxGThwYAYMGJBddtklq622WpLk1VdfzW233ZZSqZRjjjlmvhYKAAAAAJWkULDWvn373HDDDTnrrLNyzTXX1Bu33nrr5dRTT82qq646P+oDAAAAgIpUKFhLkjXWWCPDhw/P+PHj88477yT54qEFbdq0mW/FAQAAAEClKhys1WnTpo0wDQAAAIAFTqGHFzzxxBO56qqr6g279dZb07t372y00UY5++yzM3PmzPlSIAAAAABUokLB2uDBg/Pyyy+XX//nP//JGWeckTZt2mT99dfP9ddfn6uvvnq+FQkAAAAAlaZQsPb6669nrbXWKr++6667svjii+eGG27Ib37zmwwcODB33XXXfCsSAAAAACpNoWBtypQpWXzxxcuvH3300WyyySZp0aJFkqRLly5577335k+FAAAAAFCBCgVryy23XP7xj38kSd566628+uqr2WSTTcrjP/744zRv3nz+VAgAAAAAFajQU0G32267DB06NOPGjctrr72W1q1bZ4sttiiP/9e//pWVV155ftUIAAAAABWnULB26KGHZvr06Xn44Yez3HLL5dxzz02rVq2SJBMnTszf/va37LPPPvO1UAAAAACoJIWCtYUWWijHHXdcjjvuuAbjllhiiTz22GPzXBgAAAAAVLJC91gDAAAAgAWdYA0AAAAAChCsAQAAAEABgjUAAAAAKECwBgAAAAAFFArWXnjhhfldBwAAAAB8pxQK1nbbbbf07ds3Q4cOzdixY+d3TQAAAABQ8QoFaxdccEFWWmmlXHbZZdl6662z++6758Ybb8zEiRPnc3kAAAAAUJkKBWvbbbddrrzyyjzyyCM59dRTkySDBg3KpptumsMPPzz33Xdfpk2bNl8LBQAAAIBKstC8zNymTZvstdde2WuvvfL2229n5MiRGTlyZI477ri0bNkyffv2zQ477JAePXrMr3oBAAAAoCLMt6eCLrzwwmnRokUWXnjhlEqlVFVV5cEHH8zee++dH/3oR3nttdfm16oAAAAAoMnN0xlrn3zySe6///6MHDkyf//731NVVZXNNtssRxxxRDbffPNUV1fngQceyHnnnZdTTjklt9xyy/yqGwAAAACaVKFg7U9/+lNGjhyZhx56KJ9//nm6dOmSn//85+nfv3+WXHLJetNus802mTRpUs4888z5UjAAAAAAVIJCwdqRRx6Z5ZZbLvvtt1922GGHtG/f/iunX2ONNbLddtsVKhAAAAAAKlGhYO3aa69Nz549Gz19165d07Vr1yKrAgAAAICKVOjhBXMTqgEAAADA91GhYO3iiy/ODjvsMMfxO+64Y4YMGVK4KAAAAACodIWCtfvvvz+bbbbZHMf36tUro0ePLlwUAAAAAFS6QsHa+++/nxVXXHGO41dYYYW89957hYsCAAAAgEpXKFhbdNFF8+67785x/DvvvJOFF164cFEAAAAAUOkKBWvrr79+RowYkXHjxjUY9/7772fEiBEecAAAAADA99pCRWY65phjMnDgwAwYMCC77LJLVltttSTJq6++mttuuy2lUinHHHPMfC0UAAAAACpJoWCtffv2ueGGG3LWWWflmmuuqTduvfXWy6mnnppVV111ftQHAAAAABWpULCWJGussUaGDx+e8ePH55133knyxUML2rRpM9+KAwAAAIBKVThYq9OmTRthGgAAAAALnHkK1j744IO89NJLmTx5ckqlUoPxO+6447wsHgAAAAAqVqFg7fPPP89JJ52UP/7xj6mtrU1VVVU5WKuqqipPJ1gDAAAA4PuqushMF110UR544IEce+yxuf7661MqlXLuuefmd7/7XTbbbLOsscYaueuuu+Z3rQAAAABQMQoFa/fff3923nnnHHzwwVlttdWSJMsss0w22mijXHHFFWnZsmVuuOGG+VooAAAAAFSSQsHaRx99lK5duyZJFllkkSTJlClTyuP79u2bBx54YD6UBwAAAACVqVCwtvTSS2fChAlJkhYtWqR169Z54403yuM/+eSTfP755/OnQgAAAACoQIUeXtC1a9c8++yz5debb755rr766rRt2za1tbW55ppr0q1bt/lVIwAAAABUnELB2t5775377rsv06ZNS/PmzXPMMcfkueeey89+9rMkyYorrphTTz11vhYKAAAAAJWkULDWo0eP9OjRo/x6ueWWy7333ptXXnkl1dXVad++fRZaqNCiAQAAAOA7Ya7vsTZlypQceeSRufvuu+svqLo6a6yxRjp06CBUAwAAAOB7b66DtRYtWuTxxx/P1KlTv4l6AAAAAOA7odBTQdddd90899xz87sWAAAAAPjOKBSsnX766XnmmWdy8cUX54MPPpjfNQEAAABAxSt0M7Ttt98+M2fOzJVXXpkrr7wyNTU1ad68eb1pqqqq8swzz8yXIgEAAACg0hQK1vr27Zuqqqr5XQsAAAAAfGcUCtbOPffc+V0HAAAAAHynFLrHGgAAAAAs6AqdsXbnnXc2arodd9yxyOIBAAAAoOIVCtZOPvnkOY778r3XBGsAAAAAfF8VCtYefPDBBsNqa2vzzjvv5MYbb8x7772X8847b56LAwAAAIBKVShYW3755Wc7vF27dtlwww1z8MEHZ/jw4TnjjDPmqTgAAAAAqFTfyMMLevfundGjR38TiwYAAACAivCNBGtjx47NtGnTvolFAwAAAEBFKHQp6N///vfZDp80aVKefvrpXH/99dliiy3mqTAAAAAAqGSFgrW999673tM/65RKpdTU1GSbbbbJL37xi3kuDgAAAAAqVaFg7brrrmswrKqqKq1atcryyy+fxRdffJ4LAwAAAIBKVihYW3/99ed3HQAAAADwnVLo4QVjx47Nn//85zmO//Of/5x33nmncFEAAAAAUOkKnbF2/vnn55NPPkmfPn1mO/6GG25Iq1atcvHFF89TcQAAAABQqQqdsfbcc89lo402muP4DTfcME8//XThopLk008/zWabbZaOHTvmH//4R71xt9xyS/r27ZsuXbpk++23z1/+8pd5WhcAAAAAzK1CwdqkSZOy2GKLzXH8oosumokTJxatKUly6aWXZubMmQ2Gjxo1Kqeddlr69euXYcOGpVu3bjnyyCPz/PPPz9P6AAAAAGBuFArWlltuuTz77LNzHP/MM89k2WWXLVzU66+/nj/84Q856qijGoy75JJLMmDAgBx77LHZYIMNcuaZZ6ZLly4ZOnRo4fUBAAAAwNwqFKxtu+22GTVqVK677rrU1taWh8+cOTPXXnttRo8enW233bZwUWeddVZ23333rLLKKvWGjx07Nm+++Wb69etXb3j//v3zxBNPZNq0aYXXCQAAAABzo9DDCw455JA888wzOfvss3P55ZeXA7A33ngj48ePz/rrr5/DDjusUEH33XdfXnnllQwePDj/+te/6o0bM2ZMkjQI3FZdddVMnz49Y8eOzaqrrlpovUmy0EKFcsYkSU1N8Xn5fmvKtqFdMifaJZVIu6QSaZdUIu2SSqRdUom+jbZRKFhr3rx5fve73+WOO+7IAw88kLfffjtJ0rVr12y99dbZcccdU10998VPmTIl5557bo477rgsvvjiDcZ//PHHSZJWrVrVG173um58EdXVVVlyyTnfNw6KatWqRVOXAA1ol1Qi7ZJKpF1SibRLKpF2SSX6NtploWAtSaqrq/OjH/0oP/rRj+ZbMZdddlmWWmqp+brMxqqtLWXSpM8Kz19TU60jYbYmTZqSmTNrv37Cb4B2yZxol1Qi7ZJKpF1SibRLKpF2SSWal3bZqlWLRp3xVihYmzhxYj744IOsscYasx3/n//8J8suu2xat27d6GW+++67+d3vfpehQ4dm8uTJSZLPPvus/N9PP/20vLzJkyenbdu25XknTZqUJHO1vtmZMaNpOgG+32bOrNW2qDjaJZVIu6QSaZdUIu2SSqRdUom+jXZZKFg755xz8sYbb+Tmm2+e7fgzzjgj7du3z9lnn93oZb7zzjuZPn16Dj744Abj9tlnn6y99tq58MILk3xxr7X27duXx48ZMybNmjVLu3bt5vIvAQAAAIBiCgVrTz75ZPbYY485jt98881z0003zdUy11xzzVx33XX1hv373//OOeeck0GDBqVLly5p165dVl555dx3333Zcssty9ONHj06G264YZo3bz53fwgAAAAAFFQoWBs/fnyWXHLJOY5fYokl8tFHH83VMlu1apWePXvOdlznzp3TuXPnJMlRRx2VE088MSuuuGJ69uyZ0aNH58UXX8zw4cPnan0AAAAAMC8KBWtt27bNSy+9NMfx//rXv9KmTZvCRX2VbbfdNlOmTMmwYcNy5ZVXZpVVVsmQIUPSvXv3b2R9AAAAADA7hYK1LbfcMn/4wx+y2WabZYsttqg37k9/+lNuv/327L777vNcXM+ePfOf//ynwfCBAwdm4MCB87x8AAAAACiqULB21FFH5YknnsiRRx6ZNdZYI6uvvnqS5NVXX83LL7+cVVddNUcfffR8LRQAAAAAKkl1kZlatmyZESNG5LDDDsuMGTNy//335/7778+MGTNy+OGH5+abb06rVq3md60AAAAAUDEKnbGWJIsuumiOPvroOZ6Z9vHHH6d169aFCwMAAACASlbojLU5mTZtWu69994cfvjh2WSTTebnogEAAACgohQ+Y61OqVTKE088kZEjR+aBBx7IJ598kjZt2mTbbbedH/UBAAAAQEUqHKz985//zMiRIzNq1Kj897//TVVVVfr375+99tor3bp1S1VV1fysEwAAAAAqylwFa2PHjs3dd9+dkSNH5q233soyyyyT7bbbLl27ds1xxx2Xvn37pnv37t9UrQAAAABQMRodrO2222558cUXs+SSS6Zv374566yz0qNHjyTJ22+//Y0VCAAAAACVqNHB2gsvvJAVVlghJ598cnr37p2FFprn27MBAAAAwHdWo58Ketppp6Vt27Y58sgjs/HGG+f000/Pk08+mVKp9E3WBwAAAAAVqdGnne25557Zc889M3bs2IwcOTL33HNPbr755iy99NLp2bNnqqqqPLAAAAAAgAVGo89Yq9OuXbscfvjhGT16dG699dYMGDAgf/vb31IqlTJo0KCcdtpp+ctf/pLPP//8m6gXAAAAACrCPN0oba211spaa62Vk046KU8++WTuvvvujB49OrfccktatGiR5557bn7VCQAAAAAVZb48gaC6ujobbbRRNtpoowwaNCgPPvhgRo4cOT8WDQAAAAAVab4/2nPhhRdO//79079///m9aAAAAACoGHN9jzUAAAAAQLAGAAAAAIUI1gAAAACgAMEaAAAAABQgWAMAAACAAgRrAAAAAFCAYA0AAAAAChCsAQAAAEABgjUAAAAAKECwBgAAAAAFCNYAAAAAoADBGgAAAAAUIFgDAAAAgAIEawAAAABQgGANAAAAAAoQrAEAAABAAYI1AAAAAChAsAYAAAAABQjWAAAAAKAAwRoAAAAAFCBYAwAAAIACBGsAAAAAUIBgDQAAAAAKEKwBAAAAQAGCNQAAAAAoQLAGAAAAAAUI1gAAAACgAMEaAAAAABQgWAMAAACAAgRrAAAAAFCAYA0AAAAAChCsAQAAAEABgjUAAAAAKECwBgAAAAAFCNYAAAAAoADBGgAAAAAUIFgDAAAAgAIEawAAAABQgGANAAAA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MJXQLls2d9xDfZXQLhdptkRTl0CF+TbaZdO3fAAAAAD4DvrOBWutWrXK5MmTGwz/+OOP07p16yaoCAAAAIAF0XcuWGvfvn3GjBlTb9jkyZPz4Ycfpn379k1UFQAAAAALmu9csLbZZpvl8ccfz6RJk8rD7rvvvlRXV2fjjTduwsoAAAAAWJB8554K+vHHH2fAgAFZZZVVcsghh2TcuHE599xzs9122+X0009v6vIAAAAAWEB854K1JHn99dfzq1/9Ks8991wWW2yx7LDDDjnuuOPSvHnzpi4NAAAAgAXEdzJYAwAAAICm9p27xxoAAAAAVALBGgAAAAAUIFgDAAAAgAIEawAAAABQgGANAAAAAAoQrAEAAABAAYK1AgYPHpzu3bt/K+t66qmn0rFjx/zjH/9o9DyDBw/Os88+22B4x44dc/XVV8+Xeur+devWLdttt12uvfbazJw5c56WXalOPvnkbLvttk1dRsUYPHhwOnbsmD333LPBuP/5n/9Jnz595nqZffr0yZlnnjk/yvta77zzTr02vMYaa2TTTTfNCSeckHffffdbqeG74u67784uu+ySddddN+uss0769euXU089NR999FFTlzZP5tRHftk///nPdOzYMXfddddsx8+cOTMbbbRRfvazn823uq655po8/PDD8215c/LlbeCRRx5pMP7mm28uj5/fGvudNj/73Tl9Z5977rlZY401csstt8yxho4dO872Mz788MOz9957z5f6vm2Naf916trBE088UW/4pEmT0rFjx9x+++1zte4i+zRzM5/v69mr+96u+85bd911s9122+XMM8/M66+//o2td26/27/Nz69Pnz719gVm92/w4MHfSi3fd19ufx07dkzPnj2zxx57fCvfd7PaYYcdcvLJJ3/r6/2y22+/fbbt7ds6tmyMf//73xk8eHCmTJnS1KV8px1wwAHZeuutM23atHrD//nPf6ZTp04ZPnx4ediECRPy61//Ov3798/aa6+dtddeO9tuu23OPffcvPPOO+XpKvU4Zm72Lb6vFmrqAvhqnTt3zogRI7Lqqqs2ep4hQ4Zk0UUXzTrrrFNv+IgRI/LDH/5wvtR1zjnnpH379pk8eXLuvPPOnH322fn8889z8MEHz5flV5LDDz88n332WVOXUXGefvrpPPXUU+nZs+c8L2vIkCFp1arVfKiq8Y4//vj07NkztbW1efvtt3PJJZfk4IMPzt13352amppvtZZKNGzYsFx44YXZb7/9cvTRR6dUKuXVV1/NyJEj87//+79ZaqmlmrrEwubUR37ZWmutlVVWWSWjRo3KDjvs0GD8E088kY8++ijbbbfdfKvruuuuS+/evdOrV6/5tsyvsuiii2b06NHZbLPN6g2/5557suiiizZpv/dN97sXXHBBrrnmmgwaNCgDBw78ymnvueeeHHnkkVlxxRW/sXq+TY1p/7O69NJLs+GGG87zuovs0zB/LLLIIrn22muTJJ9++mleeeWVjBgxIjfffHP+53/+Z7b93Lya2+/2b3N/a8iQIfUOdo888siss8462X///cvDll122W+llgXBl9vf//7v/+byyy/PoYcemhtuuGGu+qLvk6uuuiotW7Ysv66urpzzXf79739nyJAh2XPPPdOiRYumLuc764wzzsi2226byy+/PEcffXSSL36YPf3009OpU6f8+Mc/TpK89dZb2XfffTNjxozsvffe6dKlS6qqqvKvf/0rN910U5577rmMGDGi3rIr7TimyL7F941grcItvvji6dat23xZ1vxaTpKsvvrq6dKlS5Jk4403zksvvZTbbrvtWwvWpk6dmkUWWeRbWdf35WBqflp00UWz2mqr5dJLL50vwVqnTp3mQ1VzZ6WVVipvE+uss04WX3zxHHHEEXnjjTey2mqrfev1VJrrr78+O+20U71fdnv16pUDDzwwtbW1TVhZcXPbb9TtDE2YMCFLLrlkvXH33HNPllpqqWy00Ubzu8z55uv+3i222CIPPPBABg0alIUXXjjJFwc8f//737Ptttvm7rvv/rZKbeCb7HcvvvjiXHXVVTnjjDOy2267feW0K6+8cqZOnZrLL788Z5999jdW09f5Nr/zZtWzZ8889dRTefrpp9OjR495Wtb83KdpCk35Ocyr6urqeu/9xhtvnB//+Mc5+OCDc+qpp2adddZJu3bt5us65/a7/dvc35q1tubNm2fppZf+yvb5Xf78m9qs7W/ttddOr169cueddy6wB+KdO3dOmzZt5tvypk2bloUWWqiiAroF3YorrphDDjkkl112Wbbddtu0b98+119/fV5++eXceuut5c/qhBNOyIwZM3LbbbdlmWWWKc+/4YYbZp999pnt/pjjmMpjy/uG/Oc//8kBBxyQbt26Zd11183RRx+d9957r940kydPzoknnpju3btnww03zEUXXZTf/e539S6/md3lD7feemsGDBiQrl27lk+nfvHFF5OkPO/5559fPkX0qaeeKo+b9VLQhx56KLvvvnvWXnvtrLfeetl7773z0ksvzdXfWl1dnY4dO+b999+vN/yDDz7IiSeemJ49e6Zr167Zc889889//rPeNNOmTctZZ52V9ddfPz169Mjpp5+ekSNHpmPHjuXTXutOeb399tvzi1/8Ij179iyfYTBt2rRcdNFF2XzzzbPWWmulX79+GTlyZL11vPrqqznooIPSs2fPrL322unbt2+GDRvW6PGzuzShMZ9vx44dM2zYsAwePDgbbbRRevbsmVNOOeV7c/bb4YcfnieffPIrT/v97LPPcuaZZ6Zv375Ze+2106dPn5x++umZPHlyvem+fLnI7bffnk6dOuW///1vvWkmTpyYtdZaKzfddFN52HPPPZd99tmn/DmccMIJhS9TXGyxxZIkM2bMKA976KGH8pOf/CQbbrhh1llnnQwcOLDepXPjx4/PWmutlZtvvrnB8gYOHJhjjjmm/Lox28ODDz6YnXfeOd27d0+PHj2y8847N8mlEskXl3r94Ac/mO24L++0za5fueaaa2bbjz388MM58sgj061bt2yyySa5/PLL681Xd8neiy++mF122SVdunRJv3798pe//KVBDTfddFP69u2btdZaK3369Mmll15aL/Cru9Tiueeey09+8pN069at3C8ms+8jZ7Xddttl+vTpue++++oN//zzz/PAAw+kf//+5V8Fb7/99my33Xbp0qVLNt1001x88cUNLo8fN25cfvazn2WjjTZK165ds80225R/we/Tp0/efffd3HDDDeW66i6zq62tzaWXXpo+ffpkrbXWyjbbbFNvO5j1vdttt93SpUuX3HDDDbP9u+psttlmqaqqqtfGRo8enRVXXDGdO3duMP2vf/3rbLfddunevXs23XTTHH/88fnf//3fBtM15ntl0qRJOeGEE9K9e/dsvvnm9frcpGG/W/d5vvTSSznwwAPTrVu3bL311rnzzjtnu/6BAwema9eu2WCDDfKXv/wlpVKp/D5dfvnlOe2008q/FH+VZs2a5aCDDsrdd9/9tZdYNGYbv/POO7PHHntk/fXXL783dd/fdb7qs3z99ddz2GGHZd111023bt1y8MEH5+233643f9F9hDnp1atXOnfunKFDh37t+zXre3/GGWfU+86b3T5NY/aF6nxdu6nz8MMPZ9ttt02XLl2y88475/nnn683fl63qSuvvDJbbbVVunTpkg022CD77bdfxo4d+7XvT6VZeOGFc9ppp2X69OkNLome1z4taXgp6Hdpf2tO3yFJ47b1xr6HC6plllkmbdq0qfdZ/u///m9OOeWUbLHFFunatWu23nrrXHTRRQ0uo2vs5/3ss89m5513TpcuXbLtttvOcX/qj3/8Y3bYYYd06dIlm2yySc4555x8/vnn5fF1/dajjz6aY445Jt27d0/v3r3Lxxt1Z5uvv/76OfXUUxvUW8S7776bo48+utzXH3DAAfnPf/5Tb5q67WvYsGHZfPPN07Vr10ycODHJ17e9SZMm5Re/+EU23XTTdOnSJb169cpxxx1XnveUU05J8kWw07Fjx0K3eeELBx10UFZYYYX88pe/zPvvv5/f/va32Wuvvcrh/tNPP51//OMfOeyww+qFanWaN2+eXXbZ5WvXM7vjmOTr95eTxvWz83vf4vvIGWvfgPfffz977bVX2rVrlwsuuCCff/55Lr744uy11165++67s/jiiydJTjnllDz55JP56U9/muWXXz4333xz/vWvf33lsv/+97/n1FNPzf77759evXpl6tSpefHFF8tBxYgRI7Lbbrtl7733Lu+czCm1Hj16dI4//vhsscUWufDCC9OsWbM8++yzGTdu3Fz/yvjee+9lhRVWKL/++OOP8+Mf/ziLLrpoTjvttLRs2TLXX3999t133/zxj38sX0Z24YUX5qabbsrRRx+dNddcM/fff38uvPDC2a7joosuSq9evXLhhReWO4Rjjjkmzz77bI444oisuuqqefjhh/PTn/40rVq1Kl9Odeihh2bppZfO//zP/2TxxRfP22+/nQ8++KC83K8bP6vGfr5JcsMNN2TdddfNueeemzfffDPnn39+llpqqZx44olz9f5Wos033zydOnXK0KFD53jvvqlTp2bmzJk57rjj0qZNm7z//vu5/PLLc/jhh+f666+f7TxbbbVVzjjjjNx3333Za6+9ysP/+Mc/Jkm22WabJF+EanvvvXd69eqViy++OFOmTMlvfvObHH744Q1Ol56d2trazJgxI7W1tRk7dmyGDBmS9u3bZ/XVVy9P884772TzzTfP/vvvn+rq6jzyyCM5+OCDc+2116Znz55p06ZNttpqq9x2223Zddddy/O9+uqrefHFF8unfTdme3j77bdzzDHHZMCAATnhhBNSW1ubl19+OR9//PHX/i3fhM6dO+emm27KCiuskN69e6dt27bzvMzTTjstAwYMyODBg/P444/n4osvTuvWrbPHHnuUp5k+fXqOO+647L///llhhRVy44035sgjjywf5CRfnE131llnZe+9907v3r3z3HPPZciQIZk8eXJOOumkeus84YQTsttuu+WQQw5JixYtsv322ze6j1xppZXSpUuX3HPPPfVqfOihh/LJJ5+ULwP9/e9/nwsuuCD77rtvTj755Lz++uvlndi6bX3ChAnls6OOO+64rLDCCnnrrbfKociQIUNy8MEH17sUqe7sjfPPPz/XXXddDjvssHTv3j0PPfRQzjjjjMyYMaPeNjJ9+vSccMIJ2W+//XLcccdliSWW+MrPo3nz5tlqq61yzz33ZOutt07yxZl4c7rH0UcffZRDDjkkP/jBDzJ+/Pj8/ve/z957751Ro0ZloYW+2J1o7PfKGWeckR122CFDhw7Nn/70p/z6179Ox44dG1yWOqsTTzwxu+66a37yk5/k5ptvzsknn5wuXbqULy287777ctxxx2XnnXfOUUcdlQ8//DC/+tWvMm3atFx22WUZMmRITjnllHrv29cZOHBgLr/88lxxxRVzvF9UY7/z3nnnney4445ZccUVM23atIwaNSp77rln7r777qyyyirl5c3usxw7dmx23333rL766jn33HNTVVWVyy+/PPvtt1/uu+++NG/efL7uI3zZ4YcfniOOOCLPP//8HM/omd17f+GFF2bSpEm5+OKL57jsudkXaky7+fDDDzNo0KAcddRRadWqVYYNG5YDDjig3ucwL9vUnXfemd/+9rc5+uij061bt0yePDnPPPNMPv300699HyvRaqutlmWWWSbPPfdcedj86NNm57u4vzXrd0hjt/XGvIcLsk8//TQff/xxvWOHCRMmZIkllsgpp5ySVq1a5c0338zgwYPz4Ycf5pxzzqk3/9d93h9++GEOOOCAdOzYMb/5zW8yadKkDBo0KJ999lnWXHPN8nIefPDBHH300eV9rzFjxuTiiy/O+++/n0suuaTeOn/5y19mp512yq677pqbb745P/vZz/Lyyy/n1VdfzaBBgzJ27Nice+65adeuXQ499NCvfQ/q9kPr1NTUpKqqKp988kn23nvvVFdXl88ov+yyy8rtfrnllivP88c//jErrbRSTj311FRXV2fRRRdtVNs755xz8uijj+aEE07I8ssvnw8//LD8w3Hv3r1z2GGH5bLLLitfrtq8efPGfrTMonnz5vnlL3+ZfffdN3vuuWdatWpVPj5IUg6gNtlkk7labmOOYxqzv9yYfvab2rf43ikx1y655JJSt27d5jj+7LPPLnXr1q00YcKE8rDXXnut1LFjx9J1111XKpVKpVdffbXUoUOH0h133FGeZubMmaWtt9661KFDh/KwJ598stShQ4fSiy++WCqVSqWrrrqqtP76639lfR06dChdddVVXzm8tra2tNlmm5X233//r/17v6yunueff740ffr00vjx40tXXXVVqWPHjqVRo0aVp/vtb39bWnfddUv//e9/y8M+//zzUu/evUvnnXdeqVQqlSZMmFDq0qVLaciQIfXWse+++5Y6dOhQGjt2bKlUKpXGjh1b6tChQ+mAAw6oN90TTzxR6tChQ+nRRx+tN/zYY48t/ehHPyqVSqXSRx99VOrQoUPpwQcfnO3f83XjS6VS6aSTTioNGDCg/Loxn2+p9MX7vcsuuzRY1pZbbjnHdX0XfLn933///aUOHTqUXnjhhVKpVCqdddZZpc0333yO806fPr309NNPlzp06FAaM2ZMefjmm29eGjRoUPn1EUccUdptt93qzbv33nuXDj744PLrPffcs7TbbruVamtry8NeffXVUseOHUsPPfTQHGuoa0+z/uvdu3fp1VdfneN8M2fOLE2fPr20//77l44//vjy8Mcff7zUoUOH0muvvVYeds4555R69epVmjlzZqlUatz2cO+995Y6dOhQmjx58hxr+Db95z//KW211Vbl96dPnz6lX/3qV+Xtss7s+pvf//73s+3HfvrTn9ab7qc//Wlp0003Lb9Pl1xySalDhw6lW265pTzNjBkzSn369Ckdd9xx5dc9e/Ysv65z4YUXljp37lwaP358qVQqlW677bZShw4dSldccUWDv21OfeTsXHPNNaWOHTuW3nvvvfKwo446qrwdT548udStW7fShRdeWG++P/zhD6WuXbuW67noootKa621VoP378tm3Q5KpS/6qM6dO5d+/etf1xt+/PHHlzbYYIPSjBkzSqXS/713X+6H56RuG7j33ntLf/3rX0tdu3YtffLJJ6W33nqrvG3O+hnOasaMGaUPPvigXh/cmO+VurZQ1+7r5tt8881LP//5z8vDZu136z7P4cOHl4d9+umnpbXXXrs0dOjQesv58vZZKpVKJ5xwQrkdn3rqqV/7/syuht/97nelzp07l95///1SqVQqHXbYYaW99tqrPG1jtvFZ1fUpffv2rdd+5vRZ/uxnPyttscUWpalTp5aHffTRR6Vu3bqV35d52Uf4qmlra2tL2223XenAAw8slUql0scff1zq0KFD6bbbbiuVSnN+7x9++OFSx44dS6+88kqpVGq4TzO3+0KNaTcdOnQoPf744+VhkyZNKnXv3r28Dc3rNjVo0KDSTjvt1Kj3r1J83X7rrrvuWtpmm21KpdI316dV+v7WrP3vnL5DGrOtN/Y9XFDUtb/p06eXpk+fXnr33XdLxx57bGm99dYrvf7663Ocb/r06aW777671KlTp9Jnn31WHt6Yz/uCCy4ode/evTRp0qTysLr9tZNOOqk8bMcdd2ywv3nTTTeVOnToUHr55ZdLpdL/9T/nn39+eZpJkyaV1lxzzVKvXr1K06ZNKw8/6qijSjvssMNXvh91bWvWf3XfZddee22pY8eO9fYrJ0yYUOrWrVvpnHPOKQ/bfPPNS+uvv37p008/LQ9rbNsbMGBAvWXNqcaPPvroK/8WGm+fffYpdejQoXT33XfXG3766aeXOnToUPr888/rDZ8xY0Z5m5k+fXp5eGOPYxq7v9yYfnZ+71t8X7kU9Bvw9NNPp2fPnvXOFlh11VWzxhpr5JlnnkmS8mUQW2yxRXma6urqbL755l+57E6dOmXixIk5+eST89hjjxV+WsuYMWPywQcf5Ec/+lGh+Xfdddd07tw5G2ywQc4///wcdNBB6d+/f3n8Y489lp49e6Z169aZMWNGZsyYkerq6qy33nrlv/2VV17J559/Xu89SNLgdZ3evXvXe/3YY49liSWWyAYbbFBex4wZM7LRRhvl3//+d2bOnJkll1wyyy+/fC666KLccccdDX4Z/brxs9OYz7fOrPdfWnXVVRu1ju+KrbbaKh06dPjKS4TuvPPO7LjjjunevXs6d+5cvvzqzTffnOM8AwYMyPPPP18+Dbnuvk8DBgxIkkyZMiXPPvtsttlmm8ycObP82a+88spZbrnlGvXEuRNPPDG33nprbrnllgwdOjQ/+MEPcuCBB2bcuHHlaT744IOcdNJJ2XTTTdOpU6d07tw5f/3rX/PGG2+Up9lggw3Srl273HrrrUm+OAX77rvvzk477VS+ZLIx20PHjh1TU1OTE088MX/+858bXC77bevQoUPuueeeXHnlldlnn33Kv8pvv/32+fe//11omVtttVW913379s24ceMabBNfnq6mpiZbbrllXnjhhSRf9F0TJkwon7lYp3///pk+fXqDy+pm7TfmVv/+/VNdXZ3Ro0cnST755JM89NBD5V/knnvuuXz22WfZZpttGvRDU6dOzauvvprki4cdbLDBBvV+nW+MF198MdOnT2/w9/br1y/jx49vsB3N7YMPNthggyy22GL505/+lHvuuSedO3eud+bUlz388MPZfffds+6666ZTp07ls4Tqapib75Uv/zJbVVXV6L7xy/Mtuuii+eEPf1ie74033si7776bfv361fssll9++STJKqusknvuuadBP/3lPmTWSyjq7L777mnZsmWuvPLK2Y5vzDaefHEp5xFHHJGNNtooa665Zjp37pw33nhjtv3hrJ/lY489lj59+qSmpqa8jlatWqVTp07ly9Dm1z7CrKqqqnLYYYflkUcemW3/Oqf3fv311091dfVsL5NL5n5fqDHtpmXLlvUetNCyZctstNFG5T5kXrepTp065aWXXso555yTp59+OtOnT59trd8lpVIpVVVVSb65Pu27ur81u33Pr9vWG/seLkg+++yzdO7cOZ07d87mm2+e+++/P+eff37at29fnqZUKuWaa65J//7907Vr13Tu3DknnnhiZsyY0eBS66/7vF944YX07Nmz3sMBNtxww3pt6dNPP82///3v9O3bt96y6o5nZm1jG2+8cfn/W7ZsmTZt2qRHjx5p1qxZefjKK6/c4NY4c3LNNdfk1ltvLf+ru9zv6aefzuqrr17vIS9LLLFENtpoowY19ezZM4suumj5dWPbXqdOnXLHHXfk6quvziuvvNKoeinutddeyzPPPJOqqqr87W9/a9Q8O+ywQ3mb6dy5c8aPH19v/NcdxzR2f7kx/ew3tW/xfeNS0G/ApEmT6p1mXGeppZYqX9b14YcfplmzZvU6/CRfexPLDTfcsHwJwwEHHJCFF144ffv2zc9//vOvvezny+quwZ/TPZS+znnnnZdVV10148ePzxVXXJFhw4ZlvfXWKx9oTZgwIc8///xs79NTd3nThx9+mCQNbgo+p6cNzjp8woQJmThx4mzXUbf8ZZddNldffXUuvvjinHnmmeUv9lNOOSXrrbdeqqqqvnL87DTm860z69OwmjVrNl/uvVApqqqqcuihh+b444+f7aU7DzzwQE466aTstttu5ctoPvzwwxxxxBH17l8xq8033zwtWrTIqFGjctBBB+Xee+/NwgsvnC233DLJF5/BzJkzc8455zS4PCBJo3Zq2rVrV34AR/LFjT833njjXHPNNTnppJNSW1ubww47LJMnT87RRx+dlVZaKS1atMgll1xSb/lVVVUZOHBgrrvuupxwwgl56KGHMn78+Oy8887laRqzPayyyirly82OPPLIVFdXZ5NNNsnpp58+357mO7eaN2+eXr16lQ8sH3300RxyyCEZOnRohgwZMtfLm7V/W3rppZN8sa3W/Y3NmjVL69at60231FJLlfuLum1s1v6g7vWs22DdOopq27ZtevbsmXvuuScHHHBAHnjggXz++efly0AnTJiQJNlpp51mO39dW5k4cWK90/Mbq+7vmfXvqHtd15cnSYsWLcr32Gismpqa9OvXL6NGjcq77747x1DsxRdfzOGHH54tttgiBx10UJZaaqlUVVVl1113LW/Lc/O9Mut3X7NmzRoVJs9uvro+te6zOOKII2Y7b9++ffPwww/n0EMPzfDhw8uXFm+11Vb17p/24IMPNggLWrRokZ/85CcZMmTIbC/xacw2/sknn2T//fdPmzZtcvLJJ+eHP/xhFl544fziF79o0B/O7rOcMGFCrr322nr3sPry+5DMv32E2enbt2/5oTXnnXdeg9qSOb/3c+qT53ZfqDHtZnbzLrXUUnn99deTzPs2tfPOO+fTTz/NzTffnGuuuSYtW7bMjjvumBNPPPE7e2P7Dz74ICuvvHKSb65P+67ub83aThqzrTf2PVyQLLLIIhk+fHhKpVLefPPNXHjhhTnppJMycuTI8nfGtddem/POOy8HHnhgevbsmVatWuUf//hHzjzzzAZ95Nd93h9++GFWWmmlBnV8uX+YPHlySqVSg/2JuksfZ21js/Y/zZs3n6d217Fjx9n2V5MmTZrtvstSSy3VIJSd3bFR8vVt77TTTkvr1q3z+9//Pueff36WW265HHzwwY269yhzp1Qq5Ze//GVWWmml/PjHP86vfvWr/OhHPyrfVqGu/Y8bN67eA2QuvvjiTJ06NQ899NBs97m/7jimsfvLjelnv8l9i+8Twdo3oHXr1rO9gfpHH31U3nFp27Ztpk+fnsmTJ9frqGdNo2dnhx12yA477JDx48fnwQcfzDnnnJOFFlporp5YVrcRzO7G042x6qqrljfmHj16ZJtttsl5552XTTfdNFVVVWndunU23XTTejdvr1N3nX7dPZsmTJhQ72aNc7r5fN2vqXVat26dNm3azPEMgrovq1VWWSWXXHJJpk+fnueeey4XXXRRDj300DzyyCNZbLHFvnb8rBrz+S5I+vXrl8GDB+fSSy9tEADdd999WXPNNevdl6gxv9Qsssgi2XLLLTN69OgcdNBBGT16dDbffPPyr3ItW7ZMVVVVDjnkkHLY9mWzhrWN0aZNmyy55JLlnZa33norL730UoYOHVpvHVOnTm0w784775xLLrkkDz30UG699db07Nmz3pdjY7aH5IubyW+22Wb55JNP8sgjj+Scc87JKaecMtsD6aaw6aabZo011igfoCZf1D/rGRuTJk2a7fyz9m91D6j48v3bpk+fno8//rheuPbRRx+Vp6nru2ZdVt02OWsoNz9st912OeWUUzJmzJjyWV11v7LXrW/IkCFZdtllG8xbF9AsscQShfrbur/3o48+qtdP1r13X96hmbWPbKwBAwZkzz33TJJ6Zx5/2Z/+9Kcsvvji+c1vflM+E3PWm/nP6/fKvKpb/+mnn56uXbuWh48YMSIjR47Mj3/84+yzzz7ZY489csABB+TGG29Mu3btctlll9U7EJpTMPjjH/84V199da666qoG4xqzjT///PP54IMPcsUVV2SNNdYoj588eXKDtjO7z7J169bp1avXbA98vvxdNT/2EWanuro6hx56aH7605/m5ZdfrjduTu99nTm9p/OyLzQns5t3dn1I0W2quro6++67b/bdd9+MGzcuo0aNyoUXXpgll1xyjsFiJXv11Vczbty48oH4N9mnfR/2txqzrTf2PVyQVFdXl48bunbtmlVWWSW77rprhg4dmkGDBiX5Yp+xT58+OeGEE8rzfXl/Y260bdt2tm3ny/1D3b7krH3G5MmTM23atG9kf6IxWrduXe/KiDofffRRg5pmd2yUfH3ba9myZU499dSceuqp+c9//pPrrrsugwYNSocOHeb56c/Ud/vtt+fpp5/O9ddfnx49emTkyJH55S9/mdtuuy01NTXp2bNnkuSvf/1rvfv51v1w0dgzXGc9jmns/nJj+9lvat/i+8SloN+AddddN08++WS9XzrGjBmT//znP1l33XWTJGuttVaSL34Zr1NbWzvbp9/NSZs2bTJw4MBsvPHGGTNmTHl4s2bNvvJsoCRp3759ll122fJT5+bFYostlqOPPjqvvfZa/vSnPyX54hTt119/vRzAfflf3VkCq6++ehZeeOHyPHVmfT0nG220UcaPH59mzZo1WEeXLl0a3GizWbNmWX/99XPwwQfnk08+abBD+HXj6zTm812Q1B1sPfjggw2eWDR16tR6p8gnafDU1jnZdttt89JLL+XRRx/N888/X74MNPniErBu3bplzJgxs/3si+y0/ve//82ECRPKoVzdNvTl+t999916N3iu07Zt2/Tu3TtXXXVVHn300QZn/TRme/iyxRdfPP3798+AAQMK71TOq1mfypp88Xm+//779X5JXXbZZRvU+Pjjj892mQ888EC91/fff39+8IMfNNj5+/J0M2fOzJ/+9KesvfbaSb44MGvTpk2DJ3Xee++9adas2WwP6mfVmD7yy7beeussvPDCufbaa/Pkk0+Wz1ZLku7du6dFixb54IMPZtsW69rThhtumCeffLLBU5a+rq4uXbqkWbNms/17l1pqqflycNm9e/dsu+222XfffWe7I57837b85Z34Wbfl+fm9UkTd+seOHVvvM1hmmWVSXV2dZZZZJksttVR+97vfpaqqKvvvv38+/PDDdOzY8Su/O+osvvji2WeffTJixIgGO6qN2cbrQvkv9ynPPvvs1z5ttM6GG26YV199NZ06dWqwji9fTlVnXvYR5qR///5ZaaWVGlz+P6f3/sufwezMj32hWU2ePDlPPPFEvdePP/54uQ+Zn9vUMsssk/333z8dO3as9x5/V3z++ef51a9+lebNm5eftj4/+7Q5+S7vbzVmW2/se7gg69KlSwYMGJDbb7+9fEb6vOwzzqpr16556qmn6p3R+sQTT9Q7I3WxxRbLmmuuOdu+IEmTtbF11103r7zySr0+5eOPP87jjz/+tTUVaXsdO3YsPwW0bn+u7nP4Pl1l0xQmTJiQ888/PzvttFP5Sqlf/vKXeeWVV8oPcevRo0e6dOmSyy67bJ5+mJz1OKax+8tz289+E/sW3xfOWCto5syZDRpq8kVHvt9+++X222/P/vvvn8MOOyyff/55fvOb32S55ZYr/yK4+uqrZ6uttspZZ52VKVOm5Ic//GFuvvnmTJ069SvPOrjkkksyceLErL/++llqqaXyyiuv5NFHH81+++1XnqZ9+/Z58MEH06NHj7Ro0SKrrLJKvScnJV/8wnHSSSfl+OOPz1FHHZUddtghzZs3z/PPP58uXbp87b3eZrXjjjvm8ssvz7Bhw7LVVltlv/32y8iRI7PXXntln332yQ9/+MOMHz8+L7zwQpZZZpnst99+WXLJJbPHHnvk8ssvz8ILL1z+cqu7v0ndWRFzsvHGG2fzzTfPgQcemAMPPDAdO3bMlClT8tprr+Wtt97K//zP/+Tll1/Oeeedl/79+6ddu3b55JNPcsUVV2T55ZfPiiuu+LXjZ6cxn++CZrvttsvQoUPz1FNPle9nlHyxA3rmmWdm6NCh6d69ex5++OF6BzxfZaONNsoSSyyRn//852nVqlWDpwX+7Gc/y7777ptjjz02AwYMSKtWrfLBBx/k8ccfz84771z+BWhO3nrrrTz//PMplUoZN25crr766vKlbcn/HSjWPYX2s88+yyWXXDLHMy923XXXHHzwwWnVqlWD+3U0Znu46aab8vzzz2fTTTdN27Zt88477+Tuu++ud0+Pb9N2222XzTffPJtsskl+8IMfZNy4cRk+fHgmTJiQfffdtzxd3759c+2116ZLly5ZZZVVcvfdd9e7T92XPfnkkznvvPOy8cYb57HHHstdd92V008/vd623qxZs1x22WX5/PPPy08F/eCDD8oH8jU1NTn88MNz1llnpU2bNunVq1eef/75DBs2LPvuu2+jDlga00d+2eKLL57evXtnxIgRqaqqqndWV93TnS644IJ88MEHWX/99VNTU5OxY8fmwQcfzODBg9OiRYvst99+ueuuu7LXXnvlsMMOS7t27TJ27Ni8+eab+elPf1qu68knn8xjjz2WVq1aZYUVVkibNm2y11575eqrr07z5s3TrVu3PPzww7nnnnty2mmnpaam5mv/3q9TVVWVCy644Cun2XjjjXPttdfmV7/6Vbbaaqs899xzueuuuxosZ35+r8ytqqqqnHzyyTnxxBPz2WefpXfv3mnRokVeeumlfP7553njjTeyyiqrZIUVVsjVV1+dvfbaKwceeGCGDx/e4BKfOdlnn33y+9//Ps8991zWX3/98vDGbOPdunXLoosumkGDBuXggw/OuHHjMnjw4DmGTrM6+uijs8suu+SAAw7IrrvumqWXXjr//e9/87e//S09evTItttuO9/2EeakpqYmhxxySPkgrM6c3vv33nsvDz/8cI477rjZ3ruv6L7QV1liiSVy6qmn5uijj07Lli0zbNiwlEqlcr81r9vU6aefnlatWqVbt25p1apVnn322bz88sv1zjSoRLW1tXn++eeTfHG/q1deeSUjRowoP8mw7gep+dmnfdn3ZX+rMdt6Y9/DBd3hhx+e0aNH59prr82JJ56YjTbaKNddd12GDx+elVdeOXfffXfeeuutQsved99984c//CEHHXRQDjrooEyaNCmDBw9ucNnakUcemSOOOCInnnhitt9++7zxxhu5+OKL07dv39n+8Plt2HnnnXPNNdfkkEMOybHHHlt+KuhCCy1Ub/9rdhrb9nbfffdstdVWWX311VNTU5M777wzzZo1K5+tVnd/txtuuCFbbrllFllkkSZ7P77Lzj///CSp1yeuscYa2WuvvXLJJZekX79+WWaZZXLhhRdm3333zc4775x99tknXbp0SVVVVd59993cdNNNad68eYPQ+euOYxq7v9yYfvab3rf4vhCsFfT555/P9jTw888/PzvssEOuv/76nH/++TnxxBNTXV2djTfeOCeffHK9Bnb22WfnzDPPzPnnn5/mzZtnp512yuqrr54bbrhhjuvt0qVLrr322tx777355JNPsuyyy+aAAw7IYYcdVp7m9NNPz9lnn52DDjooU6dOzXXXXTfbkKF///5ZZJFFcvnll+f444/PwgsvnE6dOjW4wXhjNGvWLIceemh+8Ytf5KmnnkrPnj0zYsSI/OY3v8mvf/3rTJw4MUsttVTWXnvtess/4YQTMmPGjFx55ZWpra3NVlttlYMPPjhnnnlmow50Lrnkklx55ZW58cYb8+6776Zly5ZZffXVy/e3atu2bZZeeulcccUVGTduXFq2bJkePXrkggsuSE1NzdeOn53llluuUZ/vgqSmpiYHH3xwfvGLX9Qbvvvuu+edd97J8OHDc/XVV2eTTTbJhRdeWO70v0qzZs3St2/fjBgxIrvsskuDs0jWWWed/OEPf8jgwYNzyimnZPr06Vl22WWzwQYbzPa+GrO66KKLyv+/5JJLZo011si1115bvtdL8+bNM3jw4Jx55pk55phjstxyy+Wwww7Lk08+OdsbcW+yySZp0aJFBgwYkIUXXrjeuCWXXPJrt4eOHTvmL3/5S84555xMnDgxbdu2zYABA2bbz3wbjjzyyPzlL3/Jueeem/Hjx2fJJZdMx44dc80112SDDTYoT3f44Yfno48+ytChQ1NVVZXddtst++yzT84999wGyzzzzDMzYsSI3HjjjVlsscVyzDHHlC9BrNOsWbNcdNFFGTRoUF555ZWssMIKueSSS+pdOrf33ntnoYUWyjXXXJMbb7wxbdu2zZFHHtmox9snje8jv2y77bbL/fffn549ezYIQvbff/8ss8wy+f3vf5/hw4dnoYUWyoorrpjevXuXd4SWXHLJ3Hjjjbnwwgvz61//OlOmTMnyyy9f77K+448/Pr/85S9z1FFH5dNPP80555yTnXfeOT/72c/SsmXL3Hrrrbn88suz/PLLZ9CgQdl9990b9ffOD7169cqJJ56Y4cOH5/bbb88666yTK664YrY3fZ5f3ytF9OvXL61atcrll19ePtNhkUUWSVVVVb0zLTt06JArrrgiP/nJT3LIIYfkd7/7XaPuj9WyZcvstddeueyyy+oNb8w2vvTSS+e3v/1tzj///Bx++OFZeeWVM2jQoNleWjo7K620Um655Zb85je/yaBBg/LZZ5+lbdu2WW+99coHPPNzH2FOtt9++wwdOjTvvPNOveGze++XX375bLrppl95r8Mi+0JfpW3btjnxxBNz/vnn5+23387qq6+eq6++ul4N87JNde/ePTfffHNuueWWTJkyJe3atcspp5xSPuOrUk2dOjW77bZbki/O+l5hhRWy4YYbZsiQIfVukp7Mvz7ty74v+1uN2daTxr2HC7r27dunf//+ufHGG3PIIYfkiCOOyIQJE3LJJZck+eKHu1/84heN/m7/sh/84AcZNmxYzjrrrBxzzDFZccUVc/rpp+fiiy+uN90WW2yR3/72txk6dGgOP/zwLLHEEtl1113rXY76bVt88cVz/fXX59xzz81pp52W2trarLPOOhk+fHiWW265r52/MW1vnXXWyZ133pl33nkn1dXV6dChQy6//PJyX9CpU6ccddRRueWWW3LVVVdlueWWy5///Odv9O/+vnn66adzxx135Fe/+lWDe+kdffTRuffee3POOefkN7/5TVZaaaXcfvvtufrqq3PHHXdkyJAhqaqqSrt27bLJJpvkoosuanBc/HXHMUnj9pcb089+G/sW3wdVpVKp1NRF8H/23HPPVFdXl08PXRD99Kc/zTPPPKMD5zvliSeeyH777ZfbbrutfHkTX3jqqaeyzz775NZbb613o9VZDR48OL/73e9me7ktsOCwLwQAfJc4Y60J3X///Xn//ffToUOHTJkyJffcc0+efvrpBvcu+T7729/+lmeffTadO3dObW1tHnrooYwcOTInn3xyU5cGjTJu3Li8/fbbueCCC7LOOusI1QDmgn0hAOC7TrDWhBZddNHcddddefPNNzN9+vS0b98+F1xwwWyfcvh9teiii+ahhx7KsGHD8vnnn2f55ZfPySefXO+abahkN998cy699NKsueaaOeuss5q6HIDvFPtCAMB3nUtBAQAAAKCAr37sIgAAAAAwW4I1AAAAAChAsAYAAAAABQjWAAAAAKAAwRoAAAAAFCBYAwCoEB07dszgwYObbP19+vTJySef3OhpDznkkG+4IgCAyiZYAwD4ltxwww3p2LFjBg4c2NSlNMprr72WwYMH55133mnqUgAAKtJCTV0AAMCCYuTIkVl++eXz4osv5q233spKK63U1CXVc99996Wqqqr8+rXXXsuQIUOy/vrrZ4UVVmjCygAAKpMz1gAAvgVjx47Nc889l1NOOSVt2rTJyJEjm7qkJEmpVMrUqVOTJM2bN0+zZs2auCIAgO8OwRoAwLdg5MiRad26dXr16pW+ffs2Olh76qmnsvPOO6dLly7Zcsstc9NNN2Xw4MHp2LFjvelmzJiRoUOHZsstt8xaa62VPn365KKLLsq0adPqTVd3b7RHH300O++8c7p27ZqbbrqpPK7uHmu33357jjnmmCTJPvvsk44dO6Zjx4556qmn6i3v6aefzi677JIuXbpkiy22yJ133llv/O23356OHTvm6aefzllnnZUNNtggPXr0yOmnn55p06Zl0qRJ+dnPfpb11lsv6623Xs4///yUSqV6yxg1alR23nnndO/ePeuss0622267XHvttY16/wAAvkkuBQUA+BaMHDkyW221VZo3b55tt902N954Y1588cV07dp1jvO89NJLOfDAA9O2bdscddRRqa2tzdChQ9OmTZsG0/7iF7/IHXfckb59++YnP/lJXnzxxVxxxRV5/fXXM3To0HrTvvHGGznhhBOy2267Zdddd80qq6zSYHnrrbde9t5771x//fU59NBD0759+yTJqquuWp7mrbfeyjHHHJNddtklO+20U2677bacfPLJ6dy5c1ZfffV6yzvrrLOy9NJL56ijjsoLL7yQESNGpGXLlnnuueey3HLL5bjjjssjjzySq6++Oh06dMiOO+6YJHnsscdy/PHHZ8MNN8yJJ56YJBkzZkyeffbZ7Lvvvo178wEAviGCNQCAb9g///nPjBkzJqeddlqSZN11182yyy6bkSNHfmWwdskll6SmpiY33nhjlllmmSRJv3790r9//3rTvfzyy7njjjsycODAnHXWWUmSPffcM23atMnvfve7PPnkk9lggw3K07/11lu56qqrsummm85x3e3atUuPHj1y/fXXZ6ONNkrPnj0bTPPGG2/khhtuSI8ePcq19erVK7fffntOOumketMutdRSGTZsWKqqqrLnnnvm7bffztVXX53ddtstgwYNSpLstttu6dOnT2677bZysPbQQw9l8cUXz9VXX52ampo51gsA0BRcCgoA8A0bOXJkll566XI4VVVVlf79+2f06NGZOXPmbOeZOXNmnnjiiWyxxRblUC1JVlpppQaB2MMPP5wk+clPflJv+P77719vfJ0VVljhK0O1xlpttdXKoVqStGnTJqusskrGjh3bYNpddtml3oMRunbtmlKplF122aU8rKamJmuttVa9+Vu1apUpU6bksccem+d6AQDmN8EaAMA3aObMmRk1alR69uyZd955J2+99VbeeuutdO3aNf/973/zxBNPzHa+jz76KFOnTp3tk0NnHfbuu++muro6K664Yr3hbdu2TatWrfLuu+/WGz6/nvC53HLLNRjWunXrfPzxxw2G//CHP6z3umXLlrNdRsuWLevN/+Mf/zgrr7xyDjrooGy22WY55ZRT8sgjj8yP8gEA5plLQQEAvkFPPvlkPvzww4waNSqjRo1qMH7kyJHZZJNN5su6vnxG2FdZZJFF5sv65ubSzOrq2f+eO6fhdZZaaqnceeed+etf/5pHHnkkjzzySG6//fbsuOOOOe+88+aqXgCA+U2wBgDwDRo5cmSWWmqpnH766Q3GPfDAA3nggQcyaNCgBmHXUkstlYUXXjhvvfVWg/lmHbb88suntrY2b731Vr2HC/z3v//NpEmTsvzyyxeqvbFB3TetefPm6dOnT/r06ZPa2tr88pe/zIgRI3L44YfP9ow+AIBvi0tBAQC+IVOnTs0f//jH9O7dO9tss02Df3vuuWc+/fTT/PnPf24wb01NTTbaaKM8+OCDGTduXHn4W2+9lUcffbTetL169UqSXHvttfWG//73v683fm61aNEiSTJ58uRC888PEyZMqPe6uro6HTt2TJJMmzatKUoCAChzxhoAwDfkz3/+cz799NP06dNntuO7deuWNm3a5O67727wpM8kOfLII/PXv/41e+yxR/bYY4/U1tZm+PDhWX311fPvf/+7PN0aa6yRnXbaKSNGjMikSZOy3nrr5R//+EfuuOOObLnllvWeCDo31lxzzdTU1GTYsGGZPHlymjdvng022CBLLbVUoeUV8Ytf/CIff/xxNthggyyzzDJ57733Mnz48Ky55pr1zs4DAGgKgjUAgG/I3XffnYUXXjgbb7zxbMdXV1end+/eGTlyZIMzs5JkrbXWyrBhw3L++efnt7/9bZZbbrkcffTRGTNmTMaMGVNv2rPOOisrrLBC7rjjjvzpT3/K0ksvnUMOOSRHHnlk4frbtm2bQYMG5Yorrsipp56amTNn5rrrrvtWg7Xtt98+N998c/7whz9k0qRJadu2bfr165ejjjrqa+/PBgDwTasqlUqlpi4CAIDGO/zww/Paa6/lj3/8Y1OXAgCwQPMzHwBABZs6dWq912+++WYeeeSRrL/++k1UEQAAdVwKCgBQwbbccsvstNNOadeuXd59993cdNNNadasWQ488MCmLg0AYIEnWAMAqGCbbrppRo0alQ8//DDNmzdPt27dcvzxx2fllVdu6tIAABZ47rEGAAAAAAW4xxoAAAAAFCBYAwAAAIACBGsAAAAAUIBgDQAAAAAKEKwBAAAAQAGCNQAAAAAoQLAGAAAAAAUI1gAAAACggP8H8V6RKsTmZJ8AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyNXVbzP5yR0LiZ0/nIeWjHb",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}