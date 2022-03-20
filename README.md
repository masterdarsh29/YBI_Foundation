# YBI_Foundation
Artificial Intelligence And Machine Learning
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.3"
    },
    "colab": {
      "name": "Stock_Market_Prediction(Petroleum)final.ipynb",
      "provenance": [],
      "collapsed_sections": [
        "vfZGT9zFbes9"
      ]
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tieXgez1berr"
      },
      "source": [
        "#  **Stock Market Prediction Using Different Machine Learning And Deep Learning Algorithms**\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2501HdlZfGAG"
      },
      "source": [
        "## **Importing the necessary libraries**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bExe6rckbers"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YoioHTGgcPVu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fa09feea-5f28-4615-83a4-15a5177865f2"
      },
      "source": [
        "cd /content/drive/My Drive/Stock Market Prediction(Mini Project)"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[Errno 2] No such file or directory: '/content/drive/My Drive/Stock Market Prediction(Mini Project)'\n",
            "/content\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c1-boyHyfakV"
      },
      "source": [
        "## **Importing the dataset**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1fMnpg-sbery",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cc663dcb-08bd-412f-e7b1-c931d8eedcfa"
      },
      "source": [
        "##Metallic Dataset\n",
        "df=pd.read_csv(\"Stock_PETR1(Petroleum)\")\n",
        "df.info()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 4087 entries, 0 to 4086\n",
            "Data columns (total 13 columns):\n",
            " #   Column        Non-Null Count  Dtype  \n",
            "---  ------        --------------  -----  \n",
            " 0   Unnamed: 0    4087 non-null   int64  \n",
            " 1   <TICKER>      4087 non-null   object \n",
            " 2   <DTYYYYMMDD>  4087 non-null   int64  \n",
            " 3   <FIRST>       4087 non-null   float64\n",
            " 4   <HIGH>        4087 non-null   float64\n",
            " 5   <LOW>         4087 non-null   float64\n",
            " 6   <CLOSE>       4087 non-null   float64\n",
            " 7   <VALUE>       4087 non-null   int64  \n",
            " 8   <VOL>         4087 non-null   int64  \n",
            " 9   <OPENINT>     4087 non-null   int64  \n",
            " 10  <PER>         4087 non-null   object \n",
            " 11  <OPEN>        4087 non-null   float64\n",
            " 12  <LAST>        4087 non-null   float64\n",
            "dtypes: float64(6), int64(5), object(2)\n",
            "memory usage: 415.2+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "F00sCC1zqlJZ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "outputId": "d0e6d019-4fe3-4bec-97cc-bcb0b2b789cd"
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-aa53753b-11f4-4f95-84f5-fdb1a19762f5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>Unnamed: 0</th>\n",
              "      <th>&lt;TICKER&gt;</th>\n",
              "      <th>&lt;DTYYYYMMDD&gt;</th>\n",
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>&lt;OPENINT&gt;</th>\n",
              "      <th>&lt;PER&gt;</th>\n",
              "      <th>&lt;OPEN&gt;</th>\n",
              "      <th>&lt;LAST&gt;</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>Petro..Inv.</td>\n",
              "      <td>20010325</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>2139.0</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>349714320</td>\n",
              "      <td>163488</td>\n",
              "      <td>15</td>\n",
              "      <td>D</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>2140.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>Petro..Inv.</td>\n",
              "      <td>20010326</td>\n",
              "      <td>2135.0</td>\n",
              "      <td>2136.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>37030936</td>\n",
              "      <td>17577</td>\n",
              "      <td>18</td>\n",
              "      <td>D</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>2100.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>Petro..Inv.</td>\n",
              "      <td>20010327</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2045.0</td>\n",
              "      <td>2050.0</td>\n",
              "      <td>200173239</td>\n",
              "      <td>97608</td>\n",
              "      <td>51</td>\n",
              "      <td>D</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2050.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>Petro..Inv.</td>\n",
              "      <td>20010328</td>\n",
              "      <td>2049.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2020.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>120265895</td>\n",
              "      <td>59019</td>\n",
              "      <td>39</td>\n",
              "      <td>D</td>\n",
              "      <td>2050.0</td>\n",
              "      <td>2100.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>Petro..Inv.</td>\n",
              "      <td>20010331</td>\n",
              "      <td>2101.0</td>\n",
              "      <td>2205.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2205.0</td>\n",
              "      <td>187171518</td>\n",
              "      <td>85296</td>\n",
              "      <td>37</td>\n",
              "      <td>D</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2205.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-aa53753b-11f4-4f95-84f5-fdb1a19762f5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-aa53753b-11f4-4f95-84f5-fdb1a19762f5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-aa53753b-11f4-4f95-84f5-fdb1a19762f5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "   Unnamed: 0     <TICKER>  <DTYYYYMMDD>  ...  <PER>  <OPEN>  <LAST>\n",
              "0           0  Petro..Inv.      20010325  ...      D  2140.0  2140.0\n",
              "1           1  Petro..Inv.      20010326  ...      D  2140.0  2100.0\n",
              "2           2  Petro..Inv.      20010327  ...      D  2100.0  2050.0\n",
              "3           3  Petro..Inv.      20010328  ...      D  2050.0  2100.0\n",
              "4           4  Petro..Inv.      20010331  ...      D  2100.0  2205.0\n",
              "\n",
              "[5 rows x 13 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9gsprJSuhGF4"
      },
      "source": [
        "## **Exploratory Data Analysis**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3KYyTRW7gcjz",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 364
        },
        "outputId": "235e03ab-56d3-456c-95d5-bd62d6252664"
      },
      "source": [
        "df.describe()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-18e511b7-9068-4688-837d-25cd0a54cf06\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>Unnamed: 0</th>\n",
              "      <th>&lt;DTYYYYMMDD&gt;</th>\n",
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>&lt;OPENINT&gt;</th>\n",
              "      <th>&lt;OPEN&gt;</th>\n",
              "      <th>&lt;LAST&gt;</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4.087000e+03</td>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4.087000e+03</td>\n",
              "      <td>4.087000e+03</td>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4087.000000</td>\n",
              "      <td>4087.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>2043.000000</td>\n",
              "      <td>2.010508e+07</td>\n",
              "      <td>2314.985075</td>\n",
              "      <td>2354.750918</td>\n",
              "      <td>2270.214583</td>\n",
              "      <td>2315.438953</td>\n",
              "      <td>8.081839e+09</td>\n",
              "      <td>2.544001e+06</td>\n",
              "      <td>307.725471</td>\n",
              "      <td>2311.123073</td>\n",
              "      <td>2312.586004</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>1179.959604</td>\n",
              "      <td>5.619828e+04</td>\n",
              "      <td>2370.609602</td>\n",
              "      <td>2423.969382</td>\n",
              "      <td>2305.411326</td>\n",
              "      <td>2366.676940</td>\n",
              "      <td>2.214442e+10</td>\n",
              "      <td>4.476661e+06</td>\n",
              "      <td>423.435746</td>\n",
              "      <td>2351.547232</td>\n",
              "      <td>2367.453351</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.001032e+07</td>\n",
              "      <td>491.000000</td>\n",
              "      <td>491.000000</td>\n",
              "      <td>491.000000</td>\n",
              "      <td>506.000000</td>\n",
              "      <td>5.710000e+03</td>\n",
              "      <td>1.000000e+01</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>506.000000</td>\n",
              "      <td>491.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>1021.500000</td>\n",
              "      <td>2.005122e+07</td>\n",
              "      <td>999.000000</td>\n",
              "      <td>1007.500000</td>\n",
              "      <td>985.000000</td>\n",
              "      <td>1003.000000</td>\n",
              "      <td>3.315275e+08</td>\n",
              "      <td>1.977020e+05</td>\n",
              "      <td>63.000000</td>\n",
              "      <td>1003.000000</td>\n",
              "      <td>993.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>2043.000000</td>\n",
              "      <td>2.011041e+07</td>\n",
              "      <td>1730.000000</td>\n",
              "      <td>1755.000000</td>\n",
              "      <td>1690.000000</td>\n",
              "      <td>1725.000000</td>\n",
              "      <td>1.476633e+09</td>\n",
              "      <td>9.672930e+05</td>\n",
              "      <td>164.000000</td>\n",
              "      <td>1724.000000</td>\n",
              "      <td>1723.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>3064.500000</td>\n",
              "      <td>2.015091e+07</td>\n",
              "      <td>2522.000000</td>\n",
              "      <td>2565.000000</td>\n",
              "      <td>2500.000000</td>\n",
              "      <td>2529.000000</td>\n",
              "      <td>5.431455e+09</td>\n",
              "      <td>2.888838e+06</td>\n",
              "      <td>360.500000</td>\n",
              "      <td>2529.000000</td>\n",
              "      <td>2527.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>4086.000000</td>\n",
              "      <td>2.020060e+07</td>\n",
              "      <td>22000.000000</td>\n",
              "      <td>22500.000000</td>\n",
              "      <td>21482.000000</td>\n",
              "      <td>21483.000000</td>\n",
              "      <td>3.370985e+11</td>\n",
              "      <td>8.097697e+07</td>\n",
              "      <td>4276.000000</td>\n",
              "      <td>21483.000000</td>\n",
              "      <td>21483.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-18e511b7-9068-4688-837d-25cd0a54cf06')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-18e511b7-9068-4688-837d-25cd0a54cf06 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-18e511b7-9068-4688-837d-25cd0a54cf06');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "        Unnamed: 0  <DTYYYYMMDD>  ...        <OPEN>        <LAST>\n",
              "count  4087.000000  4.087000e+03  ...   4087.000000   4087.000000\n",
              "mean   2043.000000  2.010508e+07  ...   2311.123073   2312.586004\n",
              "std    1179.959604  5.619828e+04  ...   2351.547232   2367.453351\n",
              "min       0.000000  2.001032e+07  ...    506.000000    491.000000\n",
              "25%    1021.500000  2.005122e+07  ...   1003.000000    993.000000\n",
              "50%    2043.000000  2.011041e+07  ...   1724.000000   1723.000000\n",
              "75%    3064.500000  2.015091e+07  ...   2529.000000   2527.000000\n",
              "max    4086.000000  2.020060e+07  ...  21483.000000  21483.000000\n",
              "\n",
              "[8 rows x 11 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mkJ2qxhUTjGJ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "outputId": "bfc42f09-1e20-4708-f4e8-4bd61daeb1a3"
      },
      "source": [
        "import seaborn as sns\n",
        "plt.figure(1 , figsize = (17 , 8))\n",
        "cor = sns.heatmap(df.corr(), annot = True)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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BzLY8PoEMleW/hEChVHmrieFoJwUMVf0E1Q5594E3vyb1fwDGOWce97MzgQ+MLPTnDv0bymZK4aDkb/nmGPq6T+lgGxvifr+oETYk/ZDuEMQkQiR+U/dS1rQ6d9rsqSC/q/cyeEOocQ5b/mT4Q5BSoHo404s1dOP73nhliJvmxw78K3D5sTfmB3unLvIv/wAgHPuqTxllgMXO+eS/cvrgLbOuS2Hqjeihj2LiIiIiIhIqTcPaGhmJ/hvOe0FjCtQZhNwAeT8zC1QEdjKYUTksGcREREREREpAmH4XV7nXLaZ3QF8DUQB7zjnlpvZo8B8/4TEg4C3zGwAOTc9XO+OMKxZjV8REREREREpUZxzk8iZyCrvumF5nq8A2h9NnWr8ioiIiIiISHCHnj+q1FHjV0RERERERIILw7Dn4qIJr0RERERERCTiqedXREREREREgnK+yBn2rJ5fERERERERiXjq+RUREREREZHgdM+viIiIiIiISOmhnl8REREREREJTj91JCIiIiIiIhFPw55FRERERERESg/1/IqIiIiIiEhw+qkjERERERERkdJDPb9SKphZuEMQEYlc+ipcCsGInPv+ROQoRNA9v2r8ioiIiIiISHARNNuzvusVERERERGRiKeeXxEREREREQkugoY9q+dXREREREREIp56fkVERERERCQoF0E/daTGr4iIiIiIiASnYc8iIiIiIiIipYd6fkVERERERCQ49fyKiIiIiIiIlB7q+RUREREREZHgXORMeKWeXxEREREREYl46vkVERERERGR4CLonl81fkVERERERCQoF0GNXw17FhERERERkYinnl8REREREREJTj2/pZuZfW9mrfIs1zezZf7n55rZhDzbLjazn8xslZn9bGafmNnx/m2jzOzKAnVnhuo4/o6OHc9h8eLvWLZsOoMH9w/Y3r59a2bPnsgff6ylR4/O+bZlZq5jzpxJzJkzibFj3w5VyMWuU6dzWbZ0OitWzOSewbcHbO/QoQ1z53zFnt0buLxHl4DtVapUZt3aebz44uOhCLdEGPLkC5zdpReX9bk13KGUGMpJIOUkUFnMSbmk1sSMfJ+Ylz+iwmXXBC0Tfea5xPx7FDEvvEulu4bkrj+mdz9inn+XmOffJbrdeaEKudidfX47vp37JdPmjefWu24M2H5T/2v5ZvZnfDVjLB9+/iYJ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OSlLOSlrv8uJ38fO15+n9n1PgsdD/sQv8a9dTcJFg/AtX0JB2jQKf5lBbGp36j43Cvx+dr71Gi5nG8TGUeehFwBwu3ay4/mHA5NfVXM+n4/773iUtz54FY/Xwwf//YRlS1Zwy53XMv+XBXz71Q/c9a9bqFWrJi+/+SQA6esyufLSmwB4f/xI2hzUilq1ajJt/jfceeNwJk+atre3rPKcz8/Cu0bS/b27Ma+Hde9OImfJOtrefgFb564k6+tZtLzyFBqf1A3n81OwJYf5N74KQO2DmtPuX5fiHJjBqlfHk7Oo+o8QKI/9rj4pB+WkYkXyGd2KZi6Sc01LSJU97Lk6KoyCDzoVbVf6j5EOQUSiRM7/6VnS0g77XpMllfZqTPtIh1DlHL/gkUiHINVAbKM21br1+HOzcyu8bXJE+kcRyUm1HfYsIiIiIiIiUl7VftiziIiIiIiIVI5oGvasnl8RERERERGJeur5FRERERERkZCi6aeO1PgVERERERGRkPz7LlJtaNiziIiIiIiIRD31/IqIiIiIiEhIjugZ9qyeXxEREREREYl66vkVERERERGRkPwu0hFUHDV+RUREREREJCS/hj2LiIiIiIiIVB/q+RUREREREZGQNOGViIiIiIiISDWinl8REREREREJyR/pACqQen5FREREREQk6qnntwr6v6RekQ6hyvERRXOsV5Cc/xsc6RCqHn2dV4Yr1L1Tmnmi59mlilL7329GOoQqZ3r/QZEOocqp3a9epEOocrYP0XVSmsWoji0t8X/fRzqEvyWanvlV41dERERERERC0rBnERERERERkWpEPb8iIiIiIiISknp+RURERERERKoR9fyKiIiIiIhISJrwSkRERERERKKeP3ravhr2LCIiIiIiItFPPb8iIiIiIiISkj+Khj2r51dERERERESinhq/IiIiIiIiEpKrhFd5mNkpZrbEzJab2Z0htl9uZllm9kvwdcW+jqlhzyIiIiIiIhJSJH7n18y8wMvAicA6YKaZjXPOLSxVdKxz7vryHlc9vyIiIiIiIlKV9ACWO+dWOufygfeAs/7uQdX4FRERERERkZD8ZhX+KofmwNpiy+uC60o7z8zmmdmHZtZiXwdV41dERERERETCxsyuMrO0Yq+r/sJhPgNaOec6AxOA0fvaIWqf+TWz+4EE59xdxdalAu8659oHl38B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\n",
            "text/plain": [
              "<Figure size 1224x576 with 2 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VWy495nCOFQ_",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "84b088ae-df7f-4c19-e320-301bd98fdf4e"
      },
      "source": [
        "df['datetime'] = pd.to_datetime(df['<DTYYYYMMDD>'], format='%Y%m%d')\n",
        "df.index = df['datetime']\n",
        "df.dropna()\n",
        "df_new = df[[\"<FIRST>\",\"<HIGH>\",\"<LOW>\",\"<CLOSE>\",\"<VALUE>\",\"<VOL>\"]]\n",
        "df_new.head()"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-f57fe7ed-c782-4ac7-9081-2d8fceac4295\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2001-03-25</th>\n",
              "      <td>2140.0</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>2139.0</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>349714320</td>\n",
              "      <td>163488</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-26</th>\n",
              "      <td>2135.0</td>\n",
              "      <td>2136.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>37030936</td>\n",
              "      <td>17577</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-27</th>\n",
              "      <td>2100.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2045.0</td>\n",
              "      <td>2050.0</td>\n",
              "      <td>200173239</td>\n",
              "      <td>97608</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-28</th>\n",
              "      <td>2049.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2020.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>120265895</td>\n",
              "      <td>59019</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-31</th>\n",
              "      <td>2101.0</td>\n",
              "      <td>2205.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2205.0</td>\n",
              "      <td>187171518</td>\n",
              "      <td>85296</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f57fe7ed-c782-4ac7-9081-2d8fceac4295')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f57fe7ed-c782-4ac7-9081-2d8fceac4295 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f57fe7ed-c782-4ac7-9081-2d8fceac4295');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>  <HIGH>   <LOW>  <CLOSE>    <VALUE>   <VOL>\n",
              "datetime                                                       \n",
              "2001-03-25   2140.0  2140.0  2139.0   2140.0  349714320  163488\n",
              "2001-03-26   2135.0  2136.0  2100.0   2100.0   37030936   17577\n",
              "2001-03-27   2100.0  2100.0  2045.0   2050.0  200173239   97608\n",
              "2001-03-28   2049.0  2100.0  2020.0   2100.0  120265895   59019\n",
              "2001-03-31   2101.0  2205.0  2100.0   2205.0  187171518   85296"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hLz42nBYber_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9454a3ee-a61c-4fef-b402-3658ef22f115"
      },
      "source": [
        "df_new.shape"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(4087, 6)"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kvkvBjzCbesH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "eac480bb-b6bb-4355-c1cc-777cd7555449"
      },
      "source": [
        "#Calculating the change in price\n",
        "df_new['change_in_price'] = df_new['<CLOSE>'].diff()"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6Q4sHYAVhXdc"
      },
      "source": [
        "## **Data Visualisation**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xvmCMSnkZO6R",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 500
        },
        "outputId": "226821db-5b54-4b4d-8c20-c170cea0032d"
      },
      "source": [
        "plt.figure(figsize=(16,8))\n",
        "plt.plot(df_new['change_in_price'], label='Change In Price')"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7efd9531ef10>]"
            ]
          },
          "metadata": {},
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA7MAAAHSCAYAAAA+KZy5AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd3xT9foH8M9JdwuFllE2Ze8hU5ayRBDce1xRcaPXn96rF9yKA+/1OlCvinvhBhd771X2pqUthdK9d9b5/ZGc05PkZLXZ/bx9+aJN0uQ0TU6+z/d5vs9XEEURRERERERERMFE4+8DICIiIiIiInIXg1kiIiIiIiIKOgxmiYiIiIiIKOgwmCUiIiIiIqKgw2CWiIiIiIiIgg6DWSIiIiIiIgo64f4+gMZq3bq1mJyc7O/DICIiIiIiIi/Yt29foSiKbawvD/pgNjk5GSkpKf4+DCIiIiIiIvICQRDOql3OMmMiIiIiIiIKOgxmiYiIiIiIKOgwmCUiIiIiIqKgw2CWiIiIiIiIgg6DWSIiIiIiIgo6DGaJiIiIiIgo6DCYJSIiIiIioqDDYJaIiIiIiIiCDoNZIiIiIiIiCjoMZomIiIiIiCjoMJglIiIiIiKioMNgloiIiIiIiIIOg1kiIiIiIiIKOgxmiYiIiIiIKOgwmCUiIiIiIqKgw2CWiIiIiIiIgg6DWSIiIiIiIgo6DGaJAsziLWeQPG85anUGfx8KEREREVHAYjBLFGA+2ZoBACiv0fn5SIiIiIiIAheDWaIAI4r+PgIiIiIiosDHYJYoUAn+PgAiIiIiosDFYJaIiIiIiIiCDoNZIiIiIiIiCjoMZokCDhfNEhERERE5w2CWKEAJXDRLRERERGQXg1kiIiIiIiIKOgxmiYiIiIiIKOgwmCUKMNxnloiIiIjIOQazRAFK4JJZIiIiIiK7GMwSBShmaImIiIiI7Av39wEQkcmCv45jcKcW/j4MIiIiIqKgwGCWKEB8ti0DANAqLhIAy4yJiIiIiBxhmTFRgGF1MRERERGRcwxmiYiIiIiIKOgwmCUiIiIiIqKgw2CWKEBxySwRERERkX0MZokCjMg9eYiIiIiInGIwS0REREREREGHwSwREREREREFHQazRAFK4EazRERERER2MZglCjBcMUtERERE5ByDWSIiIiIiIgo6DGaJiIiIiIgo6DCYJQpQXDFLRERERGQfg1miAMNtZomIiIiInGMwSxSgGNMSEREREdnHYJaIiIiIiIiCDoNZIiIiIiIiCjoMZokCjMhFs0RERERETjGYJQpQDGqJiIiIiOxjMEtERERERERBh8EsERERERERBR0Gs0QBhsXFRERERETOMZglClAMaomIiIiI7GMwS0REREREREGHwSwREREREVETsfxwDkqqtP4+DI/wSDArCMLngiDkC4JwVHFZoiAIawVBSDX/m2C+XBAEYZEgCGmCIBwWBGGY4mdmm2+fKgjCbE8cG1HQYX0xEREREXnBhdIazF2yH3OX7Pf3oXiEpzKzXwKYbnXZPADrRVHsBWC9+XsAmAGgl/n/+wF8CJiCXwAvABgNYBSAF6QAmKgp4jazRERERORJ1VoDACC3rNbPR+IZHglmRVHcAqDY6uKrAXxl/vorANcoLv9aNNkFoKUgCO0BXA5grSiKxaIolgBYC9sAmYiIiIiIiBrAYDRlS8I0gp+PxDO8uWY2SRTFHPPXuQCSzF93BHBOcbvz5svsXW5DEIT7BUFIEQQhpaCgwLNHTUREREREFIIYzDaAKIoiPLgSUBTFxaIojhBFcUSbNm08dbdEAYHVxURERETkDUaRwayr8szlwzD/m2++PBtAZ8XtOpkvs3c5UZMkMqwlIiIiIg/SmzOz4QxmnfoDgNSReDaA3xWX32nuanwxgDJzOfJqANMEQUgwN36aZr6MiIiIiIiIGslgNAIIncxsuCfuRBCE7wFMBNBaEITzMHUlXgjgJ0EQ5gA4C+Am881XALgCQBqAagB3A4AoisWCICwAsNd8u5dFUbRuKkVEREREREQNoDeEVpmxR4JZURRvtXPVFJXbigDm2rmfzwF87oljIiIiIiIionoGrpklIp/gklkiIiIi8iCDvGY2NMLA0PgtiIiIiIiIyCE9t+YhIm8SRaZkiYiIiMjzDCG2ZpbBLBERERERURPANbNE5BPMzxIRERGRJxm4zywREREREREFG2nNrIbBLBF5AzOyREREROQNf//+AABmZomIiIiIiCgISWtmT+aWo6RK6+ejaTgGs0QBik2NiYiIiMgbpMzs9He24qoPtvn5aBqOwSwREREREVETouxmfK64xo9H0jgMZokCDDOyRERERORpomKQya15iIiIiIiIKCjU6Y3y1+Ga0AgDQ+O3IApBIvsaExEREZGHKINZZmaJiIiIiIgoKNTpDPLXDGaJyCuYkSUiIiIiT2NmloiIiIiIiIJOnb4+M5tVXI3kecv9eDSewWCWKMCwmzEREREReVqtrj4zuz2t0I9H4jkMZokCFINaIiIiIvIUZWY2nGXGREREREREFAzqdNyah4iIiIiIiIKMsgFUiCRmGcwSBRpWFxMRERGRp9UqtuYxhsiAk8EsUYAKkXMMEREREQUAXahEsAoMZomIiIiIiEKcqOguKrDMmIiIiIiIiMg/GMwSBZrQqwAhIiIiogASIolZBrNEgUrkRrNERERERHYxmCUiIiIiIgpxoZgnYTBLREREREQU4sQQXMvGYJYowITiiYaIiIiIyNMYzBIFALX1saFYCkJERERE/qEcWwohsjcPg1kiIiIiIiIKOgxmiQIAs7BERERE5E2hON5kMEsUYELxRENERERE/hWKQ0wGs0QBIBRPLkREREQUmEJkySyDWSIiIiIiolCn1nA02DGYJQoAoXhyISIiIqLAxMwsERERERERBYVQTJ0wmCUKAGonFyZriYiIiMhjQnBsyWCWiIiIiIioCREQGnXGDGaJAoAyCxuCk2ZERERE5GdiCI4yGcwSERERERGFuIpavb8PweMYzBIFALWZslCcPSMiIiIi39t4Kh+vLD8hf89uxkRERERERBTwdp0p8vcheAWDWaIAYLFmlm2MiYiIiIicYjBLRERERETUhIRIlTGDWaJAxQQtEREREXlCqA4rGcwSERERERE1IUKIdIBiMEsUYEJ15oyIiIiIyJMYzBIFAJYUExEREZG3fL4tw9+H4BUMZokCFONbIiIiImqsWp0BeqPlyDI0iowZzBIFBJGhKxERERF5we8Hs/19CF7DYJYowLDkmIiIiIjIOQazRAGAASwREREReYOgUlQcKkNPBrNEAUpkhEtEREQUsmp1Bqw7nuf9B1JZIBsq40wGs0QBIDROJ0RERETkqhd+P4Z7v07B0ewyrz6OwRi6I00Gs0RERERERD6WXlgJAKjWGjx6v9/vyUJ+eS0AILesFvOXHvHo/QcSBrNEASBUSj2IiIiIyDXSdjlhHozIpOD1vq9TAADpBZWeu/MAxGCWKEAxvCUiIiIKXUY5mPVcSFanN2V5i6u1AACDnYRJZlG1xx7Tn7wezAqCkCkIwhFBEA4KgpBivixREIS1giCkmv9NMF8uCIKwSBCENEEQDguCMMzbx0cUCBi4EhERETUtcmZWUOnQ1EDS8ljpPkN4uSwA32VmJ4miOFQUxRHm7+cBWC+KYi8A683fA8AMAL3M/98P4EMfHR9RwKms1fv7EIiIiIjIS6TGTB5MzMr3eb6kBrU6g5z9DVX+KjO+GsBX5q+/AnCN4vKvRZNdAFoKgtDeHwdI5EtqFSDvbUj1/YEQERERkU8YRanM2JOZWdN96o0iHv3+gPx9qPJFMCsCWCMIwj5BEO43X5YkimKO+etcAEnmrzsCOKf42fPmyywIgnC/IAgpgiCkFBQUeOu4ifwqlNuoExERETV10lgv3AvBLACsPZ4X8uPJcB88xnhRFLMFQWgLYK0gCCeVV4qiKAqC4NazLIriYgCLAWDEiBGh/ReipoGvYiIiIqImRS4z9uCaWevglZnZRhJFMdv8bz6AZQBGAciTyofN/+abb54NoLPixzuZLyMiIiIiIgoZUqdhwZMNoIxW34d2LOvdYFYQhDhBEJpLXwOYBuAogD8AzDbfbDaA381f/wHgTnNX44sBlCnKkYlClqiSmvXkiY2IiIiIAosUeIoeyJ6KooiD50ptMrEsM26cJADLzIPycABLRFFcJQjCXgA/CYIwB8BZADeZb78CwBUA0gBUA7jby8dHRERERETkc54MNJcdyMYTPx3CveO7WVz+6PcHPPYYgcirwawoiukAhqhcXgRgisrlIoC53jwmIiIiIiIif9N7MJg9U1AJAEgz/9tU+GtrHiJSUKsuYZExERERUegymOuMGxLSGoyiRWZXMI8cQ72s2BqDWaIAxSWzRERERKGrMYHnsAVrccm/N9pcHurdi635YmseInKiaZ12iIiIiEgKZhsSf5bV6FBWo7N7n00FM7NEREREREQ+ZvBgFlWq6LPemifUMZglCgDqLdlZZ0xEREQUquoDT88FtZ4MkIMBg1miACUIQEZhFf696qRH9h8jIiIiosDh0cys+V9PdkgOBlwzSxQA1E47gzq2wD1f7kVGYRWGdm6JaQPa+fy4iIiIiMg7GrNm1oa5ztjYxIJZZmaJAkzHljEAgE4JMdAZTPUn93+zz5+HREREREQ+UKM14GxRlcu3r6zT48vtGXJEzAZQRORzyhm57NIa+TJuz0NEREQU2pTh531fp+DS/2xSvV1ZtQ6n8yosLnv+t6N48c/j2JleBKDpbc3DYJYoQBlFUd4Am4iIiIhC37a0QrvXXfO/7Zj29haLy3LKai2+ZzBLRD4nqqyaFQFoGMsSERERhTS1+NO6+WduWS0yCm3Lj2t0BgBATKSpFVJTawDFYJYoQImiCA3rjImIiIiaHOuY9OLX16verlYKZiNMYR0bQBGR76nOyIFbzRIRERGFOLUKPVfLhaVgNjzMFNZxn1kiCgiMZYmIiIiaJleD2WqtKZjNLzetnTUavXZIAYnBLFEAUDtdGa3KjE/lVqjcioiIiIiCmfqaWelfEYWVdXZ/Nr/CdN3ezBIAgL6JRbMMZokClCjCIpi9/J0tqDHPvhERERFR6Hj8x4NInrdc/l7KzH676yxGvLLO5fvhPrNE5HP2uthZ93/S6pvWbBsREbmmTm9AZZ3e34dBRA207EC2xfdSTLryaK5b99PEYlkGs0SBSu1ctCezmNlZIiKyMWvRNgx8YbW/D4OIGkAtqSFlZnPLa22vdICZWSIKCEaj7dY8932dgqd+PeynIyIiokCVml/p70MgIg8SjcCxC2VIL7DdW/aZZUfs/hyDWSLyObWW7CIAjco79PiFMu8fEBERERH5hL2teWYu2qZ6++92Z9m9LwazRCHIaBTx6db0oCrRDdVz0fLDOfj9YLbzGxIRERE1Ua5uzWOtqe0zG+7vAyDyhb+O5OCV5SeQU1aL52b19/fh2LDXACoUzV2yHwBw9dCOfj4SIiIiIv9TXzPbsPsyhmo2xA5mZqlJqDVnZMtrdH4+EteJovrJTbBucWyWX1GLfWeLvXxURERERORtDU1qNLXMLINZalIC9e1dpbKdgtr6CUdmLtqG6z/c6alDIiIiIgoJmYVVyHOzK7C/NTTB2sRiWQaz1ESYk5mB+gY/nWfbhVIUYbPPrCMFFXUePCLvS2PnTSIiIvKBiW9uwujX1vv7MNzS0DWzTQ2DWWoSpJjQ3Wynr5wrqba5LNSXPEx9a7O/D4GIiIjI71R7p/j+MIISg1lqEuR1pgF6ZtAbjDaXiRADNpNMRERERJ5x8HypzWW+bOT0t4u7+uyxPI3BLDUJ9ZnZwKRXOWHZC2TdqDwmIiIiogCk1dcnMp777ajN9b5MaIxITvDdg3kYg1lqEuTEbICmOtVm30RRdGvNLBEREREFh3u/TnF4PdfMuobBLDUJgR4UqmVmfblmtqpOjxs/2oHd6UW+e1AiIiKiJmrL6QKH1/symLW37WMwYDBLTYJgLs4NxDmusmodUs6W2Fzu6jlMFEVkFFY16hiyiquxN7MEL/xxrFH3Q0RERESNF+qNQD2FwSw1CUIAb81z95d7sCej2OZyV2fklu7PxqQ3NzXqGAzmM6b1zNz6E3lIybQ9NiIiIiJqGFeWvflyaVzw5mWBcH8fAJEvBWAsi/1Zth3sANeP9bBKBzx3SedLjdXZbM5XpvUcmQtnNvoxiIiIiAgoqtJ65DbEzCw1MYHWAKqgos7+lS4eqyfWOUhZYE0Qr5kgaqhF61Pxz58P+fswiIioiRjxyjqnt7ll8S4fHIlJMA//GMxSkyAFfIEUys5ctBXvrDtt9/rzJTWql3vjhKOXy4w9f99EgW5/VgkOZNmuW/elOr0BT/x4ECdyyv16HEREocbAxadOCUFcaMwyY2oS5LdoAJzPRFFEWY0Oxy6U49gF+wPXpQey0bpZlE+OSW8w7XXmapa3sLIO765LxXOz+iMynHNiFNyqtQbVjuK+tC21EEsPZGNI55bo1z7er8dCRBRKdAaj8xt5ycxFW9GnXXMUV2lhMIr4Zs5ovx1LqGIwS02C3AAqAKLZ3w5m4/EfXStpLK5yUIbsQdKspfWaWXsW/HUcvx+8gNHdEzFrcAcvHhmR99VoDdAb/HtuWHEkF0DgLYUgIgp2/szMOktcBIpgrsxjSoWahEAqn9h5xvW9XH11/tUb3VszKw38Oe6mUFCt1UNv9N/MvVZvxNrj5mDWb0dBRBSa/D1ZGQwCZ5TsPmZmfUQUxaDekDjYBcrWPKIoIjYysN52oijig41pAFzPzBKFkmqtwa8z9zvTi1Beqwfg/3MUEVGoUU5WGowiwnww2Cmu0mJXuuvJC2o4ZmZ94HxJNbrNX4Gl+8/7+1CaLOm05c+B4rrjeZjw7434ckdmo+7HOsvc2DmSrOJq7Dbvc+vqhItUrt2Yx/bFhwmRK6q1Buj8OHO/6mgOIsICr0kdEVEoUPZE8NX62Ts/342Hv9tv9/r2LaJ9chyuCuZ8G4NZHzidVwEA+OPQBT8fSdMVCGtm7/06xW6HYn/S6utP7O7Gl40p3w4L5jMnhZQaP2Zm9QYjVh/Lw7ierQFwzSwRkacpg1mtF4PZ03kVKDHvDXs0W32drCiKyC2rZYdlD2Iw6wPS2CRYh+6FlXUoDvqNm81ZDz+dO7x50mrsemCD4knx9dri73afRf/nV8HIkzr5ic5ghNZg9Fu3yz0ZxSiu0uKKQe398vgU3FYfy8WJnHJOghA5oFec33V6753rp729BRctWOvwNj/vO4+LX1+P/Io6uSInMATSsbgnsBbvhSjpM8bV5jqBRtrYOXPhTD8fScPVZ2b94/s9WV67b+uXlbvrs5WNETQ+nN7SGox47rejMIqAURShCeITKQWvaq0BAPy2Nc/Ko7mIiQjDxD5tAHDNLLmuVmfAA9/sAwCkvTrDz0dDFLgsy4y9c5J1dVI+JbNY/joyTAOdweCV43FXkIYoAJiZ9QmjaH99YVWdHjvOFPr4iJouaaCYX16LhStPejwjWFmnR06ZbSmxLzPb7g6GS6t18teuTrh4YsA9rEtLuVszx+/UEHqDEa8uP96o91eNOZg1GEWfZ7eMRhGrjuViUt82cmO4QNg+jALft7vOou9zq+TvlRU2Wi9mnoiCkXLS3lvvj1/2udYX56eU+ttFhDMM8wQ+iz4gvYXOFlWjqLLO/HUVCivr8MRPB3HbJ7uRV17r0n39fjAbc77cy5IiN1mHaP/4+RA+2nwGuzLsd5o7dqEM//rlMOr0trNmB7JKsD+rxObyq97bhjGvb7C5XO0+PCG/vBafbctw++c2ncpHofm1eNcXe+TL3W3K1JiZvLio+sIQI1/P1ABrj+fhk60ZWPDX8QbfR5VWL3/t6+zsvqwSFFTUYfrA9gHRpI6Cx7O/HbX4Pr+8fk/y3s+u9PXhEAW0XMUYu7HjDa3eaDOB+nPKOTz162G37ysiLHDCsCBOzDKY9QUp8EzNr8S4N0yBzqX/2YSLX1uPk7mm5lBSdsCZx344iPUn8/1WEhes6stuRby15hS2ppqy4fbOaWcKKjFz0Tb8mHIO60/k21x/7f924Lr/7bC5PL2wSvX+6nTemQn863COzWXOXhk6gxF3fbEXt3+yG4DlAD7cxTpjTwy4lR8oHMBTQ0iNPBpzPlSee33dkGPFkRxEhmswuW9buSqCp3Zy5sNNZ2wum/DvjX44ElIa/8YGfN6AyWXyroPnSjH78z3Ob+hEtVYPo1HEU78cwrAFay0q+578xTKQHbfQNqmhJjKQgtkgrjMOnGcxRG0+XSAHTgBQqwhq9EaxwYN4ZxtAp+VX4ueUc27f77ID57HsQOhuISSKwKINaU5vl1VULX/tiR1k6jxY1qI836i9Cpxl7aUBe4Y58B7cqYV8XbiLv6y8NY9Lt1ZnZCUcNZJ0HnT1daumWhHM+rIJlNEoYtXRXFzauw2aRYUHRMf1QFFUWYdNp2wnEcnkjVUn3br90ewy1Oo8Vx1UqzNgucpEalN3vqQGLzeiSoS845oPtlt8L51hN58ukCvUnMkpq0H/51fjg41pWHk0FwBQXG3Kzqq9t7JLXdu5IrAaQAUvBrNeNvvzPfhut/PmP+5OiOicRALT39liMVO080wR+jy7EmWK9ZFqHv/xEB7/8ZB7BxMEKutMv7erM0/KmzWmcZfRKOLbXWdRXuv4efclubu2+ddSNkMIV5xYHQXFjcmkSjORyswsy4ypIaR1go15j1Yryox9mZk9dL4UOWW1mDGwncXlfCsAs7/Yg7u+2OvRACxUuLvEKLOwCrPe24b/rD7lsWNYuPIk5i7Zj51n7C/TAYAZ727FhH+7lqHyhIpaHfadLXZ+Qy84U1Dpl8cl94miiN3pRZj9+R7cungXqrV6fL8ny+F7a82xPADA74cuyGMnKTP7yvKGT2CwzNgzAudZbKKkgdQti3e59SHlLDNrXXb33oZU1OmNOJJd5v5BBrklu7PkAD0uKkz1Nlq9EcsP58h/A+XguDED5TXHc/Hsb0fx+0HP7THs7GXi7FVkHQAoW9Yrf1dH4/rGjLelwFX5e3AATw0hDSYaMx6wzMz67oW48mguIsIETOmXBCC4O0l6Wlq+KTAI5Umuz7ZlIHnecreaEOaU1aDnM66thz2QVYI3V5/Cj+YKrXPF1aio1WH9iTyb2/5+MBtL95sqsk7nVeD4BfX9MZXHAZjGFY6cyCnHueIan00SPfTtflz/4U5U1emd39gNtToDTpmXhKlZuPIkpvx3s0cfk7ynsk6PmxfvAmBa/rfgrxOYv/QItqXZNmPt8+xKfLjpDGrME2tavVEeG0n/5pS61vNGTXgABbPBjM+inxVWmsoUcspq3VorpQxARFF0Wh4nfZj4cusVR45dKMP5kmrnN2wgncEIg1HEr/vO4+llR+TL7QWV/9uUhrlL9mPtcdMHvWVQ14j1eF7ILFjuC2vL2eHKrwXB8nvA8nd1ZQDSkDGK9BgWa2bdvxsi+b3gbuMypWo/rJkVRRErjuRgXM/WaBETAaB+j2c296vn632vfelVczbn/m/24f0NqapZ6MPnS/H0siPILavF0ewyjHl9g8uv0Wv/twPvb0yT19d2TIjBzR/vwpyvUnCuuP6z9/eD2Xjsh4N44qdDqNUZMO3tLbhi0VaH973anKXaYSczm19Riy+2168dXbQ+FfkVtTiR4zhIbqzj5vuv1hrwxI8HccDcpLFWZ8DrK0+gooEVUv/3w0Fc/s4WlFRp8fB3+5Bp1Rvjo822a5j3nS3Bj3ttq/KKq7QNqtSav/QIkuctd/vnAs2p3ApsS/XvDh7WyZ6CClMwWqM14MmfD2FHWiFyy2qRWViFOr0Rb6w6Kfc9qdYa5HO0NIbRNOLzJzKAyoyDeUKV+8wGEINRdHlQtvRANh68tAcA4PPtmVjw13GkPDsVrZtFWdzOaBSh0Qjymy6sEa/WN1adxNVDO6Bvu/gG3wdgKnuauWib6WsX9q7NLatFVnE1RnVLBGDas/VgVineuGGwxe3eWXcaVXV6PDOzPyb+ZxMiwgRkFrkWMF8wr29YfSwP0wa0s1gne/83+xq8x25kmHomuDGczeQ7W3NnlINZc2bWTjDrKIiXrjI0YOBdP6vJMmNqHOvXckPUKMqMfbVm9tiFcpwvqcHfJ/eSL5PXzDaBt4LBKKJKq0d8dITq9dJzEKrrh9cdz5PPg+tO5GHdiTzkV9Qhq7gaN43ojCsGtceW0wW409y0Jr2gEg9N7Nmox/xie6b8tfJ1/tgPB+Wvn//dskMyABw6V4qPt5zBrMEdUFWnx40jOqve/6L1qTCKIv5vam9c8/52XCirz1Z9tTMTn2/LQIU5Y7r4b8MxqW9b1OoMaG7nNSCpqNUhOiLMbjmmKIpYfSwP3+zKlDvMZhVXY+mBbGw8lY8Dz0/Dkt1Z+HhzOiLDNPjHtD5YduA83tuQhltHdsG9E7o5XX606phpjeSwV9ZCFIFtqYWYM747Hptqev9eN6wjlu7Plm9frdXj+g9NDSJvHtnF4r6GLViLqHANTr1i2hd406l8ZBVX484xyRa3K63WoqJWj86JsQDq96qXxnSNkV5QiWqtAQM7tnB+4wYQRVM/GLXjvPydLQBcG/s5k5pXgfc3puGVawZi55kitG8Rg3Ml1fg55Rw+v2skBEHAkfO21YjWE0LKPiI/7zuPn1W22FHuSCEqxjBVdXpsPl3Q4N8hoMqMgziYDZxnkWAwijh4rhTJ85bj8PlSh7dduLK+AYTUsEmt1OGN1Sfl+waAH/eew+srTmDgC6vlUi41F1QWr3+46Qxu/GgnRFHE/KWHse+s7dY0J3PLMeD5Vcgts1928fM+x42pzhVXW2Qnpr29GTd9vFP+fv7SI3LplNI761LxydYM6AxGZJfWuBzIAkBFrelD9ldzqZV1QsBRtuSOT3dj0pubVK/zxuJ+QyMzmtJARgoglSf2goo6VNXpkVlYhaxiR8+fVCrs3hEYLQJnxb2F5piVvEyaiGlMA6gqP2RmVxzJQZhGwGX9k+TL6vuth/xs3ygAACAASURBVL4X/ziGwS+usbtlmfQcBPN5wWgUkTxvOd5ee9rm8nu/TrG5/a70Imw6VYCHvzOtRb1T0X11V7pn14Gm5Vdi+jtb8JbVsSn3v3zwm33o99wqXP3Bdqw4kouHv9uPJ385bFEVBpga3TzwTQreWnsa76xLxepjuRaBLGDay7xCUfp7/zf78NC3+zDoxTXYkVaIrKJqrDqag/SCSqTmVchbGALAoBfX4IFv9tn9XTadKsCD3+7D9rT6LLEUSErJASnIlbr1Lz+cg/SCKry64gR2pttf93umoNIiGyq9Hstr9Xh7Xf1zZ71Ny+It6RbfF1XW4a01p7DiiKlplrIh5F1f7MXzvx+zeeyJb26SO1QPX7BWvnzDycY3Rpv8382Y9d42DH5xNfJd3BbSHQ9+uw9Xvr/N4rLzJdUW59fkecuRPG+503XXjry3IQ2/H7yAXenFuP8b02M+/N1+bDxVgJ3pRfjvmlM2xwHYLtPLMb9eX19pv7GaNG4yGI3y2GnCvzfiqV8PN2rf2vAAyswGM2ZmvcjdF7hBFOX1LIu3pOPvU3qhd1JzAMBPKeeQEBtpcXtRFCEIQn15msow6OPN6Zg/o58cOCw9UD97+MOeLDw7q7/qsYxduEF15qyiVo8anQHf7zmHZQeycXLBDIvrv911FlVaAy5+fb3dmbfDKjNlkj0Zxbjp451488YhuGF4J5zOq0B5rXvrX6pd3OZI6fIB7bDyaC4uH2AaXFqf7Kq0Bsz+fA+evqIfhnVpaXGd2joLiTc2xHa2XtrZAPCHvaaJgCqtATd9vNOi697ezBLMem+b3OlYUlBRB6MoIik+2uIx3B38W27Ho/41kavql094psxY74MW26IoYuXRXIzp3goJcfXndCk71BTeCsvMn0NavRFR4SrVKyrVG8FGWt7y7vpURIQJeHPNaSy5dzRu+3S36u1P59VPLqtNJntiaxHJ/ebg8KSDdaBSNtKatDRK8tryE3LZMQA8uuSA08ePjw7HOvOWd/aej1tHdUbHljEAHAdwuxwEo1LFhlTWGx8TDlEUVSfi1aRkOp5EkKrprDviWr9s5y09Ii9hkhzNLnOYGS01N+vcllqIIkWw7GhnhKyiajy8ZB++vmc0EuMi7d5OUl6rx5bUQtwwvBPKa3VoFhne6KwvAIvXg+lxdBj/xkbcOqqLzW1XHMnBmB6tXLrflMxibDpVgH9M643D58vwxyHTsjG1sfZtn6i/rgDbcYuj90H9z5j+1RtFeTwtikBaXuMafwVUZjaIl3UEzrMYgpRdMl2hfIP9dTgH0942lWPkl9fiqV8O4z6r2Vxpj0Xr8rQ8lZk2tUFBQ7eLOZhVar5P2+tUByYOjqWwsg7vrDstZ+xO5prWvBw6V4qiyjr5OQBM6xn2ZNR/uNgrt21IYCRNBEjPifXJ7viFcuw7W4Jnlh2xuE75WMo1T9Ll3jg1uNry3R7luiHl8ymxDmQBYOSr6zD6tfU2l7sfzKp/HcRjVvIjTyyfsCwz9v4L8VReBTIKqzBjkGUX4/rMbOi/GVw9RwfrM1Gnt9y65s01piyevcDNmqvbhfiD9bpT67ee1oVSfVcmqL/fc05+3pR2pRfhFcX2N472mJYys1LlVfPoCGQUVqFEsatDWbXOJtssjS2W7HFcRSZl6worLAN864qsSpXf9/eD2XhrjXqHaeVnvNp6XGsbTuYhu7QGi7eewdHscvx5SL03SGZhlc262/jocFTW6TH4xTVyJV9DGYwiFq2vbwomvY6lsZFUKq3kauycll+BGz7aifc3pqGsRoe9iomGuUv2u3WcDZm0lHq8VFj9LRs7ARpI+8wGcSzLYNab3M0Q2gsMHv/poOrltVrbN1G1Vu9y0OFs24MsO2W6Z83lp2oDkigXMpHK2Z+nfjmMd9alYp+5UYM0mAzTCKi1Crb/s/qURbmx3ijiu91nUVBh+cHvTtMl6Uik85HUBfH5PyzXDknPlSAIFgNe5b7Byg8g6el2lkX1BmfjRE90z2toGaC9dbLBOmgl/5LGoI1pAOXrMuMVR3IhCMC0/lbBrLTdg+IQ9AYjymoat63X+xtS8bfPdmOjG+WJtToD0vIrsfNMkVe2x3H1WRYDeC9qncGo2jXXaBQxbuEGi7Jadzkqd/Q36wmfxqxXd9cnW9Jxy+Jd+HRbBsqqdbj9012qk6+S+mDW9B4K1wg2WdmHvtsvL9vafLoAyfOWo/vTK3CmoBKHzjle7nUytwKn8yqQa5VAWK94r4miqNpbQm8UVfe8zyqqxriF9dsZVVq9jqwnu0RRxD1fpuC6/22Xe3Sorf3PLavFOpVO1vExEXJZ98oj6tl4V/V4eoVF6fqIV9aZj8f+O15tvfLMRVvxwUbL5yYt3/Lv7Oo2i2oaMi5bc9z2uQMa/5kRSJnZYMZn0YvcfZGr3f4fPx3CyRz1EojPt2fg2IX6kl0RpuyltbT8ChxTabVvvcjdep3rJf/ZqPq485eayqfqSy1E/JxyzqZk7BUXNg8vNW86XV+2ajoJh2tsCx6KqiyD1leXH8czy45i7neWs3LKANMZ6RmXPmyq6kzP37liy+ynlLEVYLnHr3KgqVxjKn2YOJo19oSGZKGHdGrp/EZOyOtt3V0za1FarH45kas80U2yxmJrHu9HT6uO5mBUciLaNLds1icoSmx2pxched5y9HxmJYa8tAaXvbUZo15dh6LKOhw+X4qyGh02nsxHWY3O4YD7r8MX8Oaa09iaWoi7v9yLrakF8n6YNVoDHv/xoFzSWlajQ/K85Rj92jr0fW4Vpr61Gbd+ssuisY0jRZV1OJ3nvFwPUHY0V79eGrC/ueYUdpxRX8YhiqJb29pIiqu08t9cFEVsTS2QM3NGo4g6vcHcwEa0e37VGYzo9cxKDHhhNe7/OgWpeRUordZi55ki3Pt1ik0pbiixHqf4IpaNjTSNK15dcUK+bMOpPGxPK3JYgiwFs9Ihn8gtx/4s2xLjteYgT1nKfdSFbQyv+WC7RfWY5EBW/XtSb1R/nVoHVNJtpG2PJNbBrDXp+rzyOnk8JWXH/7vmlNw5+OLX1+OV5Sdsfj5cI8jjG1eSEe4a9MJq6BSJiegIy8f4amcmbllcn6QwGkUcu1Busy+y8hQvivVjxYbw5LissdU8gbRmNnCOxH1cM+tFDclaWb+Y5IZEKt5dn4qiqrr67KIoqu6F9tWOsy49/hur3JsNNhhFlNfqsPV0IZ785TCe/OUwOrSIlq//dFsG7hnfDTvPFOH64Z1U70M6Edz08U4cfnEaaszZ5uiIMJsPTesMwVc7Tb9XfoVlEK4W0DsjfZDYy+pKazIEwfJDSBnMvruuvrxmV3oRJvZp67OGMkpqZYqiKKJKa0CzqHA0j2742/7L7Rl48c/j6NEmDkBjy4zVA1tPq9UZ8MmWdNw6ugtaN4tCal4FurWOw9bUQozt2Uq1NL5Ob4DeICIuiqfIQCa9FxtTZqxcDuKN9+sX2zPwc8p5fDNnFEqqdTidV4mXrhqgeltBME2w3frJLovLU83N+uYu2a/aDGjJfaPRKi4Kfdo1t7jcep3e3z4zDdbfvHEIdAYjlh3IRnREGOaM74apb5n2ySyu0uLxqb2RFB+FeUuP4ExBJZ5ZdgRPXNYbFbV6JMVHIyYyDBW1OjSLCocgCDAaRQw3Z2GseyXoDUZoDUbERta/l6T3+7ELZWgRE4FebZvDYBQRExmGvw5fkD8Xvtl1Ft/sMp3nL+6eiB/uH4N1x/Ow8VQ+jKKpbPHYS5e79T4dtmAtBndqgTdvHCIHIq9fNwjHLpTh212mMsjureNwoawGvdo2x5+Pjkd+eS2ufH8bOrSMwad3jpD7DgCmjM2a43lo0zzKpkooFBVaTSqfdaPRYkMpXzuSOJXLrFk3Ofx4czp6JzVD+xbRctMfwDSRn17QuLWP9uzJKIZO5bxiXa6tNRgRrQmzyTg62zNXWlsbGxkm90SRxivvbUjDe0jDzvmT7f68UQSeXWaqRIuO8PzuCxV1eotS3PjoCNTq6l9DomhqcPbZtgwM75pg0cyvsk6PZlHh2HQqH78oki8XLViLuZN6NPiYPHmeb+yyr4AqMw5iATdSEwRhOoB3AYQB+FQUxYV+PqQGs5dtsjf7n1FYpVp24khJtU4e6Cxan4pWcVE2t3HUiEpqIgWod951lvnbeDLfoszHuovhzYt34lxxDWYObq96olR2s/x0a4Z84m4RE2HzPNnLuFqfl9xpk14/EWD6194sqHScgmD54aIMZg8qMiTLD+dgYp+2PmkoY03tT/bt7iw899tRbHlyUqMCxxf/NGXbzxSY/uYHskpxw/BOFn/bT7emo1NCLFo1i8RrK07gi7tGoqW5eZm9ANbT6wQPny/FsgPZeH5Wf3y1IxP/XXsaZTU63DkmGZe9vQUXd0/ErvRi3DU2GS+qBBaPfX8Qq47lYtnDY3FRlwSPHht5jlQZ8P7GNERHaPCIYqsbV1VbZGZdfx2qbaVWVq3DfV+nYPbYZMwY2A7dn14hX7c3swRp+abJxssHWJYYSwSY3hfhGo3q2kN7wZLU7GTrU5PkrTxWH8u1u6/2P38+hDnjuwEAsoqr5EAWAB6a2BOPTe2FWp0B85Yewfa0QpzMrUBptQ7Lzd1Yt8+bjHELN+Clqwage5s4OUgGLD9TAODer1Ow6VQBMhfOxKFzpejRtpmcCbrtk92ICtdgQId47M8qxWezR+AROw2EdqUX4+21p/GuYk0eYAq+pWB26f7zeO63o3jtukE4ml2GT7Zm4OcHx2BkciLKqnUY8vIaAKYmhMqMWmFFnRzIAkC6+TPtSHYZlu4/j5yyWuSV1yGvvE4O2q01hUAWAO7+Yq/F9wedlOJ6QmFlnc1YxHrtohpp7KMMkE7nVeLaizrKTcgAU+XV5P9utvhZT02w3v7pbjmzrPSb1XuzVmdAdESYzRpS6/W2jyw5gFmDO8hrXxf/bTgAoFlUuHwu+3JHJrq1jpN/5l+/HrF7fAajiD3m9aeCYNofd3hX9z/zHAWIR7PrKwNLq9WXTSxQqeTbm1GMSX3b4i6r1xwA5Jc3/P3mj3GZPYFUZtyY0m1/C5xnEYAgCGEAPgAwA0B/ALcKgqDebjcI2AtmrdvhS7Y76IprT0mVVj5hbzpVgBqd7QlebRsbifIEpJahsvfBLanTG/HOulS710vluvZOdOdL6me1Fq1PxWfbMgAAMZFhNqUg9tZuZRVXY4fiuXM3wwzUD4qr7QSzUiBdp7NcwyaV9VhbczwPWr3RL2tm1TZkf+4308zr6bwKj5b0/rr/vM22Ca8sP4EHv92HT7ak40BWKY5fKEd2aQ2OZpdZrIFzlpn949AFPPbDAbljoTvu+TIFX2zPREm1Tl5/lltei2Lz30vKbqXbWW8ldfI8YafE35O2pxXi5o934vs9WRBFU5nj3CX7sc7OGh1/Kqiow+zP9+DbXWflklRfq9MbUKszlYKWK96L1oNDV1VrDfLs+LkS9SxTWbUOPZ9egTlf7oXBKOKDjWno8fQKm61lvtiRgT2ZxZi7ZL+8HENiFEWsOJKL4V0T0E5RwaIkCAJE839qnJXHVdTqIYoiPt2ajge/tb+dCQD5XKvc0gQAHp5oynhIayGlSUXlpKW0pm/DScuMCQB0m79CXoO476yp+ygAbDldgKs/2I6BL6y2uH2d3oj95rLM/21y3OzGOpAFTBmtWp0BeoMRT/x0CFVaAx774SA+2Wr6/W78aCcGv7jaYkmOtf/a+UwGgCd+OoTvdrlW3UTeY13Oa53ZVCONjZLM77fu5gDPlWDNusy1MVzpn1KnqP5SUlt7/ffv6yd8pH2ClT9XWq2z2D/Y3rgGAN7bUP+eOny+DNd/uAMlVe6XyZfYGQsBwP/9WH8srjQIk9z95V67zZ1+3ncevZOauX6ACoE08RQRHjgBZOAcifsCLTM7CkCaKIrpACAIwg8ArgbgfPFlALI37vjQzgf2e25mZQFTZjY8TJAHOSvcXMD//sY0dGgZg6uHdkCkynoJ6/3TrD31y2GXHkcZ1CkHavZO8pV1eps1vLtVOu9KXO0SaY8061tl53he/su0D1xqfiVmvVe/b5m9YKisRoftZwr9UmZ8yb83IvXVK+TvlcdQozNYbBcUESY0es3H5tMF2HAyD5P7JllcfsS85sgo1g9+h3Sq34rAOphdtD4VxVVaOVMqfWD/fvACrhrSwa1jiok0/Y7KPewMRtFmQmSLSha/TDFzrDMYYTCK+O1ANqYNSELz6AiHjyt3sXZjhvN282t3d0axRQC0/HAOTi6YblPRUFBRh4TYCJtGXqIoYsFfJxAXFYZrLuqI7q3jPD7T+vrKE9h8ukCufsgrr8MNH+7APeO74dLebeQM2dmiKkSFh9kN2hqjz7OrAACLbr0IX+7IlC93tGx2R1qhfI74be44fLYtQ+74mRAbIQ8E31x9CjeN6Gzz8yuO5kBvFLH+ZD7+PHQBS3absnjj39iIEV0TMLRzS7y+8iRaN6vfDsN6EvFccTWO55Tj2Zn97B6nlJm1954sdDIIq6jV4fnfj+GbXWcxfUA7vH3zUPR7fpXDn7Emvd6kbJY0kWe9nAOwXwXz0eYzSIyLwGsr6icWX/zDdi9Na65umaJUVadH3+dWYayD7T3Ka/X4+w/Ot4yxx7riyBmp5Lh5VHijGkFRPakCTSJVCTlSUq3D5tMF8jKi9MIqaATgoi7O+0Y0tnzUXV/uyMS/pve1O25UUk7wSkuj8hxkKg+dt58936Gyz+tFC9aiTfMovHbtIIu9sB3xVoCo7ApuLblVnMWWVq5SWzvsL4GUmQ1mgRbMdgSgHAGcBzDaT8fSaL5oalNarXXp5GePlFX9cNMZi43KPe2DTWl4+gr7gzhrC33UzfG2T3fj0ztH2N1uR2KvxFntODsnxqC0Soflh3PQr308AOCRST3x/kb3JysawnogrCydrtUZLNbMThvQDlcMbO92a3tr93yZgsyFMy0yVdKapDs+q59oOKTYY9i6AZRUsdC1VazNB2PyvOX4++SeGNuzNRauPInZY7vi2ots12HvPFOEOz7bLf89lb9Xan4l1hyzzXa+tz4Vy4/k4OcHx6B5dITFzPGejGK8YB6ETzzcBh1bxuD7PVkwisC0/kmIDNfgL5UP22n9k5BZVIWJfdpCEIA/Dl5ARJgGPds2w9gerXDX2GQ5GNUI9ie++j63CjMHtUenhBh8vCVdvrxzYgzyyupwUZeWKKnWolVcFHYq9ltUToyN7dEKeeW1GN41AS1i6oNxQRDQIiYCJVVabD9ThPE9W6FKa4BWb0SLmAicNq8tzi+vw66MIkzq01benkAp5WwJUsxByP2XdMfqY7nyOroXr+yPrOIaVGv1EAQBUeEaRIQJaBkbibIaHcI1gsVyAo0gIDEuEltTCxEVrsH5khpc0rs1ACAhLhKdE2Ll2yqzE3dc3AXf7srCYz8cQLfWcWgZE4EwjYAureKwbP95i6ztNR9stzj+kmodbh7RGT+mnEN+RR0WbzmDUd1aYbl57Wbz6HCLZjDKLENBRR1WHs3FyqOmSURHjX+kbP/0geolxoAp8+ooO2lvsk1y6ye7YBSBBy7pjn9N7+t2Y6xxPesDQulnpUG9u02NlIEsYH/ir7Gkwa7aoFzJl02ZbhvVBY9f1ttmGxTyvdmf75H3jweAzomxaB7leFLSHz7cdAaX9U+yaJbkKQ2ZsC6oqMN9X6fYrIG3xx/bSSUrSqmDVSCtmQ3iKuOAC2ZdIgjC/QDuB4AuXWw3YQ4UvghmS6q1HtnXJKespsH7zrpi8ZZ0t4JZX7rXav9ed7ohqxEg4LL+SRbld2N7tvJZMGvNei9cZfv9Ns2iMHNwe8xd4pnHcmcbEeX747eD9euXXrIz475oQ5q8pvzgj6W4ZmhHm8zj6mO5drPhafmVSDPP7g/u1AKHzYG1VGKYcrYEk/q0tTguaY0gALlcUmKvVT9gyljV6Y02s8ZZxdXYcDIfSfHRuHJIB2QVVVsEsh1aRONCWS2aRYXLkxDKY5BI5fv11Qr2Z6elQf6ZgiqLtVsGo2jxnj+RY9vxfGtqffm+co2ZPYsVATdgmz0RBNOHd53eiKhw078xEWHyh6jeKNqs8T+VV4HYyDC7VRxDOrWQG8fZWyPqzEMTe8iZVOsgzB0tYyPsrgk7kFWKwZ1aoJMiIPc0o2iaQLhrXDf5sr9P7ulyL4aLu9nPbgaqj61ec/7Uq20zpOZXujQovG5YR5c7RVPjrFZMYvZOah6wg/aSKq1Hts7zpJIqLRLiIp3ersgPHby7JHrvXOorgdXNOHCOxV2BFsxmA1DWeHUyX2ZBFMXFABYDwIgRIwJ2Tw9PrjG/akgHFFdpsc1qXW1jAy8AmNovCZ/OHgFRFNFt/grnPxDknrmin0WLf2ulNY07KY/p3grTBiTJnQUB1zoveotyTiUtv9Ki/FFZFukJ+63KBAd0iEefpOZ444bBiAjTYEdaIeZ8lYIBHeLl/YoB4N+rbNcn7Xl6Cv722R6cyqvAa9cOwoaTeVh3on4bhjq9UbWpWIuYCBx6YZrFZWNeX2/RvfKru0fheE455i7ZLwcfu9OLbYJZV7WMjcCfj4zHplP5uHFEZ0RHhKFOb8DmUwXQGow4lVuBK4d0QJfEWPR9bpW8PcrTy0xlxe3io7Fz/mSL4Dy/vBajrPaMvmlEJ/yUUj9JIgi26423PjUJnRJikF9Rh+MXyjGyWyJqdQa0iou0Cf6rtXosWp+GjzafQddWsRadSRPjIu0uM8hcOBPzlx7G93vOYcE1A/Hl9gy5KdjeZ6airEaLqW+Z1rjNGtweb944BJV1egjm+5U6a0tNTySiKOJMQaX8s5LjL0/Hv345jB9TzqF1syikPDvV4vovt2fgVwfFBUM6tcDvj4zHnZ/vsSgtT3/tCjkDeVn/JIvuvyOTE7A3U73s9dmZ/TCxTxv0bNvc5jpH2bgZA9vbP0gPuW10V4vvbxrZ2SaYHdSxhbwMQGn5kRw8OsX9Jlqe9OTlfbD+RJ68jjZQvXTVALlqQ3L5gHZIzU9zae/VKwd3YDDrB10SYxu1lZc3afVGr3b2d1e4RsCs97bhwzuGYbCTLf0cNRr1lou7B9/kmzWWGXtGoD2LewH0EgShmyAIkQBuAfCHn4+pwTyZmU3JLEYzL2wTMntMV3w6ewSAwOhkNqhjC4vv/zmtt8cfw9mvWVLlenbR2nu3XoSXrxmA8b1aW1zujZb3gGltmjPKV6G0nZFEWuO49alJdn++ZaxrJVm/H8zGg99aRhSPTOqJt24eKp+wx/ZsjRMLpqOLSimxNUEQEGXek65f++b4dPZIvHvLUNwy0jTf5UpTDYlyU/vrh3VCy9gIjOvZGgefrw96pfGNvZLfF6/sj8gwDb6+ZxQ6JcTg87tG4KM7hqF5dDh2zZ+Czomx+NuYZPlvHRUehmkD2mHW4A74x7Q+6J3UHNERYWgXHy1nVqWJk18eGmPz/msbH420V2fI3695/BL8+4YhGNAhXr7savNa4jGKD/X4mAgIgoCk+GhM6tsWzaLC0bpZlOr7OzYyHJP7tgVgmtRSeumqAap/+ycv7wMAaN8iBgDQMiYCn80eidevG4SM169Am+ZR6Nm2OY6/fDnum9ANs8eanpPWzaLQynwc0rnM+n0hCAI6tIyxuGyYeX3b0zP7oVNCDK4Zart+WrqfPkm2wSUA+f341d0jcej5adg5fzIOPn+ZxaD2kztH4MiL9a+Hqf2SkKD4/T+6YxjuGdcNvzw4BvdO6K4ayALAv6b3Vb3cdJ9t7V4HAOueuNTh9a6w7n2gts7b3jkw3smacDUnXp5u9zqpmZRkmAtrFUd1S0SrZrZd+Rsr4/UrnN/IDYM6tbC5TN732IWP0qiIQBt+NQ3hYa7nn+JUuhB7k9Zg9Ope1+N6tkJSvOvvrV8fGgsAuOHDnfhhT5bD27q757wz+5+7zOlterZtWAOoQBJIwWwAhAANFjjPIgBRFPUAHgGwGsAJAD+Joui8a0SA8uR7+4lpfdC1teslFW2aW56w+rePV71dXzuXN8TSh8dafN+QEpA/Hx1v8b00YHZmar8km4GTPdbPjbXGZGavHNIBUeFhqp2hPU0URbvrwJSNjxxtrySVNHVOjLXZzFxir2xSKSE2AkfO22Z64mPUB8f2ujwrB+KCYFqDCZgaPQDA1UM7yp0one2/p6TMjL967UCLwO6BS02PIXV3lZqFPDuzH/58xPR6jArX4K5x3XD61Rm4pHcbbPvXZEzum4TpA9vjyIuXuzVZ0TkxRu6aGyYIuLR3G7ulp+FhGiy5bzTWPXEJepsDtQTzNkcjkxPwxg2DseTe0fjojuFIe3UG9jw9xWJdrCtGdUvEjnmT8ezMfhbB1KzB7fHJnaaJro4tY7Ds4bHIXDgTcyf1BAA8eGkP/OeGwZg5qD2SW8fh1lFdLJ7X2MhwPDOzP0YmJ7p1PLGR4dj99BQcf/ly7Hl6ijygahETgY3/nIhnZ9k2uJ/SLwkdWkTjjRsGW1w+tV8SvpkzCk9ebgowBUFAi9gItG8RI28XpaScMOzfIR6PmbOUd1zcBdMHtsfzV/bHCCe/z0MTe8iTctOttuDp2srxGi/pnNnLPEibNdh5JleaXLhySAf89MAYm+vVJkFvsLPvdwsXJ64AU2fYIy9OQ0xkmEVjNyUpEy39Xu/echGuvaij3W6ycyf1wIiuCR5pnPfVPaPQLt7UgMy66kFpYEfbz8CP7hiG92+7yOH9R6k0TBxtnlRy9hoBvDfBSY6FCYJLmXPAIyu43FKnN7rc7Xdsj1bYMc/+/rFqhndJwO6npzqd2Pls9gj8fXJPDOncEn8+Oh6juydi3tIjeOqXQ3Z3lfB0s8sEJ+eiM695dnLKnvho71bVufpa9ITbRnfBMqtxeqgIf2WW4AAAIABJREFUtDJjiKK4AkBI1LqqrR8clZwo7+nljt5JzdC6WSQ+3ux4fVBcZBiuM2eepCYwkWEa3DSik8X6tU4JMThfUmMzQ7n87+Ox/HCO3IRkdLdEh12ElS7q3BLt4qPlLFjL2Ahkqfyoq0H+mO6tbNbx3jKys8WG9ZJR3RIwfUB7p1s7AMAlvdrg0ztH2KyVlZS5ELy548u7R3p8H1UADtc4788qwRhzd09Hj1yn+GBqzHoJvVFU7YZtL7DaY+c19ccj4zD9na0ATCf5WYM7YNZgy0xcrDkwdScz++jknnh95Uk8cGl3m0Hk/Bn98EvKeRRVaXG2qL5JTWJcJAZ1aoFd86eoDlwbqnNCLHaZmzWJcD4bOraHZZZfypbePLILosLDMLZn/fVt4xvWPVjKhipnugVBwMjkRLsNQCLDNbhRpfOvJySZf49Yq/J8e7PYbZpHYcf8KQCAnx4Yg/YtouX9Vt0hCAIm922LTgkxmNCrDSb0aoM7xyS7XZb460NjYRRFREeE4aeUc3jql8NIiI1QfY8oRYZrkLlwJkRRxKm8Cmj1RtUGY0oPXdoDd47pirjIcNXjDNMICNcI0GgEuRTwlpFdsPZ4nsWaaMA2E7XkvtHyHrYA8Nej4zHrvW02+y//8tBY1OmNeGvNadw7oRsulNagSmvAoE4tcOj5aYiPCUdZjQ4tYyPx9s1DkV9ei3/8fAjPz+qPzacLkBgXiVN5FfjntD4QBAFt3MzMRoVr8NjUXjiWXY6HJvZAdEQYerZthu3zJqNGZ5AD+p8eGIP1J/Jw3yXdERGmwdHsMozr2dqmNHxKvySLs6Hyc61zYgzOFdcgUWUd4aW92+DIi9Ocdj0H1CcZqGGGdm7p8n634RrBbubcUTM+X9AbRIQJlgcwsU8bNI+OkLuvA6Zx0MLrB1v/uEOf3jkCE8zN9JxV4U3pl4Qp5kqdxLhIfHn3KLyz7jTe25CGYxfK8dEdw23Or57uEePsGK33+PaWmMgwlDvZ03hcz1Y2W5y5ypcV78/O7GfzmaoUxInZwAtmQ8m6E7YNYhy94S/p3cZiPdfto7vg9tFd0SI2Ah1bxmBbquN9aB+b0guPX2Yqy1Vug3D61RkWP3vlkA4Y0TUBL/xxzKYb3IAOLTCgQwsUVWrxY8o5XD20I45ml1l00Xz75iF45a8TKDKvp5O2IRAEyw8JR79rx5YxyC6tQZhGkGf0rEvsvr//Ynyl2HoDMJ1YJH88Mg5XvW/qTKpRlKQ6oxEETHXQbr60RofWzSIxtkdr1T1O7x6XjLzyWpe3QRrWNQHZJZ5v8++oHMliKyQHnzGtFGtmGzNBqNUbVbPR9soWlXvSNY8OR4X5w0I5S2nvJB8bZXqcaq3tB4y9LPT9l3THvRO62/0AXPHYBFz6n43475rT8ntIOhZPby/TKTEWOQez5fVR7s7MSplZV/ZZbIpGdXMvE2zt87tGWnzfkPV1yqD1phGdMa1/kluDL0EQ0LddvOprHDCdg6YPaIdureOg0QhOg6cTC6bjVG6FvK1YZLgG38wZjf1ZJZj73X6sfGwCvtieibvHJVv8nDLg2vP0FLSNj1ad3IgI0yAiTIPnrzRlzZWl4lK2V5kJbxsfjW/mmDYq6KVSGv7CVf0RHxOO5NZxeGbZUYvr7hnXDU9N74PoiDDsPFNkLmtXLzcM0wgWv8OobokWr49x5omgq4d2wK70Inxw2zB0ToxFRJgGoihiQq/WuGpIB1w/rBM2nsqHIJgmQ4uqtDYl+B/cNgyAelm3tV3zp/ikQaQ7WjeLbHTH58en9kafds2d7nHsaRd1cT2YDdNobAKlXx8ai95JzdA8OgIjX11nsQTm14fG4voPdzi8z4cm9rC75aIj+56dij8PXcAbq06hRmdAtVYPjWA5LP9s9kgIgEUw66g09ZFJPbEtrRAHz5XinnHd8Pl2U8WRvTHPtP5J2JtZjBLzJP43c0bZ3CZMI+Af0/pgaOeWePzHg5j13ja8c/NQTOpbv2zCUWY2OkJj09/F+rJ7x3fDp+bqKGvK3hCfzR5hdzLcG1ypsvt2zugG95tpSFAujZ/d5TRhEcTRbECVGYeaJEWWpJs5aHS0rkCajb64eyI+umM4Xr12EPp3iEdH88DA2YteeX6eM76bxXVSZ9SxPVrh7ZuG4M4xXbHysQl2F9BLmUSNAGyfN9miNHfmoA54xrxX4l+PjsfGf07E5icnmo+h/iAcfVZ3aGkaFClLRdQGJNYB2z+n9cGzM/sh/bUrMLhTS7mB0e2ju8otzps7Kwtxtma2WgtRtPzAmNCrNW4fbeqcndwqDn+7ONnxnaC+VDBCo/F4UwdRFO2W6gKWHfIcZYWvGdpR/tr6aVn3xCXy1kKOaARTllgt6xQfo/63UGaVldlbZbddeyfeOCeZWbXYUBAEh++fpPhozBnfDX8cuoDD5j35vNUkpHNCDEQRuFBaA6Mouv35IZVflddw/8pg0TI20qUgx1psZDj6tmuO9i2icdvo+s79L1w5AKO7t3I5Ex8RplF9/Q/rkoCd86egZWwkHr+st03ptTTR0ioussFZ/4aQStRvH90Vx166HIeen4bB5lLmmYPby9UVY3q08si6uXdvuQi7n56KEcmJ8ue2IAj4Zs5o3DiiMzQaAVP6JWFy3ySEh2mQFB+NqPAwPDalF766ZxQyF87ETBdKwiXtWkQ7zdI78so1Axv8swDw7i1D5a/7tmuOzIUz0aed+hpwV7VpHoXHpvay2HrqZg9Wbkyw6kOh1NmNDuHhYbaZ2dbN6t+fXRUZRwHA8K4JSH11BvraeX6+vHukw3XygP218K2aReGucd1w5MVpiDRvRSaNeZJbxWJinzYIM1dVPGJe3gE4DmYfnNgDV5p7KRhFEb/PHYcvrCbolDq0jMGmf07CqORELLh6ACb0amP3tlP6JeGvRyegY8sY3PPVXry19rQcxEr/Ksvzn7y8D6b0bYvPZts+/o55U+Sv77i4Cx6b2gv7n7sMPz84Bn88Mg5A/d9c+Tq6tHcbzHdxZwzrHizuGtezld0xjCTl2amN6jcT3oBxxvie9t8LTRUzs15097hkVNTq8PCknjiVW4HrP9zhsIRFek1fe1FH1b0InbXwVmZ4pBKQ7uYgOs9cItWtdZy8TtJRoCLFkBqNaU/ISX3ayF1UNQJw3bBOuG5Y/borafZbozjHzhjYDscu2G734cyvD41BZqFpTaH1/mhxUeG4d0J3+fs9T0+Vj1Nvbh+tPDmM7dHKZv9BR+eOqHANyqp1MIqWZbMf/204qrUGlFbrcN2wjjiZW2Hxc/Nm9LVowgMAS+67GHsyii2yyZ5iFAGdg3bZFmWxDl5zjk7CPds2R5+kZqpbtig1j45AWY1ONWh2ZQAvvW6HdWlpsXZUsPN5LQW87qyZdcUDl/bAd7uz8N81p83H5dG7l0nvzXMl1RBF9zPic8Z3x5nCKvxtTFfnN6agt+LvE2AURRRXa7Fkt+MmLI40ZLwlvTf92aREalJXH4wHTkZTquJoCE8uXXD/ses/k6RxxROX9cGp3BSn2dlHJ/fEHRd3xWirTutqZdNhHtx2xCiK+POR8fjryAWb5VaX9U/CiZxy/KzYDs/eNlkalTWzyomexXeOwJLdZ/HmmtPyK82Uqbc9pttHd8HEPo6buv316HiniYjwMA3G9miFjafy0TkxGQDw+9zxFmvY/zGtN7QGIxZvSUdEuP37i4sMk4PAGQPbYUhnx03X5s3oa1oO8aDtens1XVrFYunDY/Hsb0exaH0qDp0rxTs3D5WD2Yl92uLS3m2w+XQBBnZsIfdYuHNMV3y766w8Bk6Mi0Ryq1hkFlXjlWsGyfefGFdfOfH2zUPx4h/H8PTMfrhheCcsP5Lj1tZFThMbTsyf0Q/zlx6xe31spKmxoT3fzhmNOz7bbfd6oP785g6Nl04dwbw1DzOzXhQdEYanpvdFs6hw9EoyzR4/dGl3u7eXTnj2qkedZmatvj/+8uVY+X8TANR381TO7jsilexKH7gvXVU/E+zoOKQPiZHJCXh4Yk/8+lD9CdJRIyKl4V0Tcb25QYkyM6s2k60xz1oCpg/TRyb1xPf3Xyxfr7Y+wFEAJ30AigAiFB/EsZGmjrAf3D4MzaMjbJ6DO8d0tfnQaNM8Sp6t9/SaWb3R6DAzq3yuXH1kteflN/O+ncoOutakmUu1P68rJTSCABx76XKLvxtgv/xWCmZr7DSiaKj46Ag8MqknssxbBnmrMYMczBbXmNfMuvc4LWIj8MFtw1TX7FHo0WgEhIdp0DqucR1+G/J6lgZNjgbPvuL/I3Cf8jz898mmQb30udKYzGxj1enrz51h5j/y8K4JSHnWeQdZjblTujWpzFrJ1ayT9Dl53UUdkW6u1hrVLdGi5NVoNHWQloKjp6b3kUu9w8ME/OfGITj20uXy7e19PoZpbCd2whXRQWJcpMVEvaRaZzt52r2N/aqAf18/GK9fNwgDO7awqDiSWF82pV8SzhZV46R58tj6PScIglyVFGknoEuINXWy751kyraPdrB1zW9zx2HrU5Ma1IgsOiIM/7lhMF67dhB2ninCrPe24ag5cREmCHLFX8eW9a+Tl68eaNO0adOTk+z2ZACA1s2i8P5twxAfHYERyYl44coBbh1nY9fVOuvurhzzSLssKLnSObp9ixi50aKrvLXzCLsZk1Px0RHIXDgT01X2GZS6yErrd/q1Vy9ncfbBYP1CjI0Ml2dge7RphsyFMzGgg2tlF/+a3hcPT+yBmYNMx6vMLjp6I0kDpoXXD4ZGI2B410T8wzx7rW9AZwUpmJ05uL3D7WOk4/rn5X3Qt108PrlzBB64tLtqVtTRs9gyJlIuM3Y0+LP+W0Q7WVfh6TJjvUF0uGZWmUmx99hqH7D2tG0eZbeZ02X9/r+9O4+XpKzvPf59qqq7zzL7MBuzwcCwDjAMwyoMIKuQOCAGUAwoKhgFUYiiMTFuJJhXjMuNei8hGr2aGC9uuGI0xjUoxh2JgogCIipLlJnxrM/9o6r6VPfpfXm6q+rzfr14cU51n+7qnjp96le/5QmrCOIrs2uXjDY9Ubumai3L8VIwrzel3rsfX8ncNdHbYFaSnnXcRu0d9cj2K5hdvWhEBd9Emdn2y4z77d+vO1nvvXx+3xQGq9uy905+Or5SX+/k2aV4onknmYxB+cKfniIpnIdx7ZkH6oNXHl9eA7vWe/qdV5+h4zbNZaaaTXTtVLLyq93DatOK2hO5914yP8Ct9Rm6df0SfefVlUHz2y4+Uu9/3rH6u4u2yvOMvvFnp+m9lx+jkzav0Pdec6aWjRd1zenh34xFIwXdfcNT9Ccn71duk4mPieSxUS+Q8T1vXgaqejfjv5/Jv6O7a/y9WZlov3ryQZUZ2guPXq9nHBMmD2oF/+973rEV38fLdn02Wus6qJF+i//mJ/fr8ifNtZQdtbH1eQFb1y/paEhezBijZx67Qf8vyujGPb2eJ125Y5O+/PJT5y1fZozRxuVjeuMFh817vE7tXTXTIpmwqVXh2I4Ny1t/fzo9xzMmLJ1u62c6e6pUB6vNDP4vFPTtvzhTd73ubP3hEXvr239xRsWUyKTmPbO9O1IXjxb08rMPaqukI9yH8P/JLGwhCmzqBV9rGgzYiZd2uOa0zTX/INRzxiGr9MqnHKyR6LmTPcSNgpQlYwU9vicsM270dlb/W7hehH161s4rwU5K7k69rPC/PL8yE1rrFZx1aDg0ot5yRvfdeK62RmtHxpnSmy/brq9e/2R9tcGyAc8+YZ+6t8Xq/TvFFyhqDoBq+qiNjRR8XXvmgdHX/fl49L1wLdX7H+2szLjfNq1YoB1t/nHF8Ovk70N8garoYKmxZt5w/hb9w6XbddDq3i0n129rl4zq09ecpJujzMsx+y4rVwvV+vcoBp6WjIYVF+uXjeo/X3mafvSGs/XdvzxTn3zx3LJ13Xxm/NX5h1VcOJxtcpH5xqcdpn9+/rH65+cdq49fdaKeGvVjnhBNy9++canecN6Wmktd1Tpned3OQ7VkrKjzj1yrg1Yv1A9ee5Z8z5Qv5kvhgLA4Y7hopKBv/cUZFfM9Cn44xOm6Mw/Qp685qWYGreB7evezj543qOvJB62c9zex+sJC/NmfXFprV9Xfm7c/c1vF0lnJwXFn1Bi2tH5Z5TKDS6verzWLR3Xo3ovKq2AUapRo1wpmX/2Hh+gbrwr7Ty8dQOtJvHzPSZv30pKxggqeJ88zdQPlL77sVF10dGsVgq34xItPKlfFHbBqgW5I9JM/owfP02iwVfI46rT6zqj93+d27r+6jfPmITsVaUt6LnFmWDJ7uLRB6WDySt2mFeO699e7tHy8WJ4q7HK9qno27bVA9/56V2VPTvQHLQ6+qq9gfeLqE/XQ//y+5uMdsveihmUozcRXajcmrrDFb9PmlQt096+eqLj/krFC2K9rG/cP1Lpq6tLMrC33CNeS/Pytd8WwlcPlkDWLddudD2vFwlLd+49GJx17ooFM48Wg6Vq+rRyr9e4yVoiD2ToDoJo+cmMXbFurlQtL5aWN+mH90jHd/9geWdmh+L1FOhywaoF+/PATze9YQyfX2+KT52IPex87NVYMagYJw66VIXpSOIhptODrhvO3aNOKcV135oHlYLAU+Fo82t0wm9j2fcKL5f942XY99z3frFktlnTxMbUDgnMPX6Ov/eQRbV61QM86rjKI2rZhiR58fE/NC9WHrwsDjzdftHXebe1aOFLQwWtqZ69LgadTD1qpa07brNd+/Ifaun6JPvqicLDQY9E50z7Lx/R/n3vsvPOuhSMFfeX6UysuoFdP42008Ou0gxr30Upzay8nnX7wKt35i9+q4JuaFzsmy8Fs5W0rF9aeMu7KsvGi3nv5MdozNeP8wv6y8aJe+ZSDdPFNt5eX9lq1qKSHfzshzzN6xyXb9N37H9f/+VLjZS1j1XNWfthgZkixhQq4pkz75+7t9LYuHAn0y/ZH16QOwewAPPuEfbRoJNDbonVgWxX/YVuxsFT+JVqWCGafcUx/1nxsx5svOkJfv/fRiqty8VXE6URmNvnLuHxBScvbXFewVS85fbN2T07rgm3r9OqP3Vlx24deeIJ+87sJPflNX5QUZhmvv+V7emz345q1tuHJX7u9GL0vM27cM5tc9qHeveplyq/csancs/zrJ8KLDI3WfSz3sLax7mtyuFNySmPFfeq8xYHvqRR4866U94oxpu+ZyfXLRvXZOx/WsvHi0GVmMbxuverEhu0FjXSSmZ2skQlC7x2xbrE+/MInyRij5QtKenmd6bhvuWirrGxba2wn7bWgqAOipZBOO3iVbr3qSdpS1Xr0iatP1Afu+Lned/vPdUuDoUDx8kvrakwS/vALw6BxZtbqDZ+8q7z9lU9pPPW3l+KAL86AJv9kLx0v6vXnbdFpB62sWEYqqfp1tbMGba3f0R2bV+j9X/+5Pnjl8SoGtaeLn37wKr3183fXvVi+Y/MKvfur9+nofbpbfqwfjDEN1zDtp+M2Ldc3//z08jCmz1yzQ7+Nlq4757A1OuewNU2D2Qu2rdOHvvWAXn/eFj30+PzkysFrFunKHZv0kn/9Tnlbctm++NAoBl55Pe9WmA7GLvXrnKFfvbguEMwOwGueGjaxtxvMxhnO2VmrZx23UX/+0R/okL0XlbOLnTTy99rCkcK89cziaYmd9Mx2a8lYUX/z9CMqtsVXwRaNFOaVJy0ZD8uMfWMafmAke2ZbGXLR+wFQjXtmk6Vj9QZvzStTjl7GBUetK5/wxE+xZslo3YA8Pu7iMuNWPg+Td/mjOss3NLpaOVb02wqeh826pWN6ZNekSsH8/i2gnpGC3/HnfCcJk7jqono9cvTGJ64+USsXlbRyYWulgOcdGfaIvv/rP6t5e+CZhn9nP3XNSRXfx1nSpC1rF+sNaw+rmDBby6kHrtR7Lz+mXG5cS3XAdnnVkoH98J1Xn6EXvO+/9Ofnhmsex1nX6n354+PaK8n9+NUn6vZ7H9X2jUt13yO7Gt53ssaF5tc89VC94OT9Gvapblm7SKsWler+bTv1oJX679efPRTnesMmOVV46XixYZVjLW+68Ai96cLwXHG/xGCv1+08VK/+2J36h0uP0rqlY9o1OV1e+zqZuY/Pj/7gsDX68LcfLG9PrpFbi2cqz5neevFWXfOB79T/AbWXyW0nPk1xLEswmybxh/F0FMw+67iN+rt/+3H59mE9EAte455Z1xq9T0tGi5qcnpVpUvqRzJLf8arTe72LTYVlxo0ys3Nf1/sgrffvkfybf/3ZB2r9slGdcXD98r7RJmW/tZ+j+cHaOJgN+jIAypX4hOah3/5e24b09xbZ0kk5+5a1i/WOS7bp1CbLj6AzW7pcB1MKy1XjCeyBPxfMnrd17/I0+hULS/rd76daDppb1UoFy303nqvHdk1qwUjgJMO/ZKyoD1wxl1GOh2h1m3U6dO/F5QGazZa7Ga0RbBZ8r+nAJWOMdm5dqy/+6Nd170Mg25n7bjxXr/34nXr3V++bd9ttL9lR9+cuPX4fXXr8PuXvn3nMBh2xbon+4H99RVfu2K+8vZywSBxmxkg//etz9bqP/1Dv+upPaz6+MZUl5c2Weooft1V5uVhOMJsicYYzmXVLTjsc1oM2HjHfaGCRSw2nGUfvp7XhHd94wWHauHx+ViIOZlstH+51mfHUzGzjzGwLT7hf1bICpsZXS8aKeuEptcuAY6PlpXJaL/ttqWe2wW1jRb/2AKjhOMSaWr80LG0LB0AN5+8tsqnWeqCNnHNY455KDFby4mPyIuaMlT70JyfoBw/+jy45dsNAV+dtN0vW0+euUWbcL5+7dofed/vPdeH2+Uv7tOrlZx2oa7tYvxj11To/+JfnH6cDV9deQaQWY4y2rF08r0c5HrS0rOYgtPqPt3xB5f3rrRpRsQ99OtdP85kIweyAtTN6Pw6gkoFKcq3Jfp8Tf/qak/TDX7TfSR73f0wPSWa24TTjxAeJZ0zdqXtzpcWDOUWYmbUNe2ZnKsqM599ea1hEHFTV+6PfbABUnJltqcy4y/uMlYL6A6BSEBwmr9AP/94iSxaN8Gc/7U7Yb27yb3LgTrJX75A1i3TUxqU6amPt1RHyol6ZcT/sv3JhuY2sU4HvaQiGh2fe5687WZv2Gu/Z+cJLTj9AB65eqNWLRnTzV8Is7GjUQ9zonLM6qdCKfpUOp+DUqS6mOgzQR1/0JN320vrlDdXi8pJkj2Gtcfj9cvCaReXBQO2IBzGUpxkP9Brx/F/Y0YKvI6PlZZLvZ6Pf6/gDsOXMbDs72ILpJtOMk/sVD0KI1VsnMH5fWi1HPG9ruERD9TTjVrQWzNa/03idzGxaLB8vlt+3NP8BQXo8MRH+vixq4co/htu+e42XhyntNV57ON+5ZNQlhX8rCr5hajwqEkH7rVjQ0wvfxcDTzq1r9Wg0kFUKl+aS6i/dWO9crBmO5Pm4RDtAW5v0XVRbNFLQ1//stIrJspVlxsOpnJlNBl8D3NnqD7C7Xn92+evkmnSNPufiz6ZWynml+kOYOjU903id2anEe33dB79bcdt7Lz+m4WO38kf/sy/doU3RUJi5MuM4M9tdP+yHX3iCbvvBLxv+/FjR1y8en2p4n2FmjNH6ZaP68cNPcJIFJw5YtVAXH71eV568X/M7Y+g998R9tdeCko7auFSn/O1/6MT999JvnpjQf//yd5KktUtrT+nNG2OMlowVU1Gxg/5y0YZU6/fO78Gxt+OAFfrSj8Ne6k4frvnPpfd3hGA2ZVZVLYCcXHh7WD+sC8FwDYBqJBnMNgoyypnZFh/3kL0X6fhNy3Xh0ev00n/9bvMfaGJ6tvHSPMlys5892njyYrVmh9FXrj+1YtmCgm/ke6atzGyj93bbhqXatqFxadxYMSgHz2m1fumYfvzwEyn+84E08T2jGy84fNC7gR4JfK9cKRW3jXz3/se18+1f1Tsv2eakrDYtlo0VnfTMYri5qAqMl7p6QeKiYd1quA6fo51z/fbu28neDAeC2ZRbkoLMbMEbjgFQzz5hn6YB19IWy4zjtVWfdmRrZdelwNe/XHGcJOn8I9dpn1d8sqWfq6dRmXHgmYpgtnpYQL0Pt3hrvZKYeGv1WnLGGI0W5pbKaeU47PZYHS/52jVRYwDUgEvY21Humx3WX1wAqXLE+iU15yHk3bVnHqDxAa2BiuHhYnVIzzP66V+fU3Gedf6Ra/XZOx/WZ+5sXHHWSLK6j1OG+fjtTrkFpaDp2nKDFvjxAKjB7mMrgxlGCr5KgaeJ6dmGV7RGCr5+8NqzNNbhmPzTD16pz931q45+VgrLZepdHGh30e5qnVzBHin4emz3ZPM7RuK3dv2yzkrhRgsNBkB19IjurYvKkSgzBoD+OevQ1YPeBQyBDU2WRuqV6nNHY4ze+axt+vLdv9Gl7/pGR4+534oF+vLdv6n5+A33pY3nSPOZCAOgUi7uBwm/HvDO1BEvKTTVYGDRMFlSXpeu8f0WlIK6Wcxmbr7saL1u51xwffahq3XWofXXcq1mra07HboYeJpM3NbqcTE3zbj91zRa9MoTlFsd7nTzpdt1ywtOaPu5pDAzu3tyuue9yC7Fmdkh/bUFACAznrZt7cCe2xijg6qWAGpnSNvCxBT6jntmM3y2QTCbAcvGe7MoeL8UvOHIzLZqyWi8Ll1/38/kv9fLzj5QJx/QfLHsmJU0VScbX/S9qjLjdveryXPXCCBrLRLfzOmHrJrXA96qsWKgWStNdJGBHrT1Ud/xkP7aAgCQGcMUzK1cWNJLTm99PeHkntfLoVQHy1K7S/MMz/vTLoLZDHC5PE8nCkHcMxsGHsOeTCtnZvv8PNUT7tr5HLG2/rq9nZYZm/L/6/TMNtjB0UQ/kos/GHHPcq2+2bSIS6wpMwYAoL8GPQQsXhbtT07ZT/+p39qVAAAgAElEQVTxslM6ruyrd451xLr5K6TMDHELYi/RM5sByeV5hlG8NE9ymvEwn77HwWynHzSt8qsuJbUT5FvZupnuYuBpoqLMuP/v9mhh7sW4iM3iYHb35IyWJ7YP+4WSpIUjBa1YWCqvHw0AAPpj0JnHkYLf+YC2xL638zLipbqyjsxsBiwdKw51qWIh6plNW5lxvyU/WJP/fBcfvb75D9v6PcjVZcat70/80O3/O3VSZtyN8VJ4Ha7WEKhh/l2o9p7nHKMXnsq6nwAA9FOKTg3mMXW+rrhPkxfY9PZ2dmjIEMxmwMpFIxoJhje7U55mnJYBUFEPcr/LP7sqM1b9iwOljqcZN96B684M+zsWjsyvBBgtzh1/Lj4Q4+fbNZneMmMpXH945cLO+oYBAEBr0nShu6E6LyRNlWm9RplxBjz3Sfvq1ANXDHo36hqWdWZbFWdm+/3B53VxKclaabbOJ1cx8DQxPZexrH4ZTcvS6/wzXXLsRl1y7Maat7kulY3XDGy2bjAAAMAwDYDqRr864NIc7JOZzYDFYwUduWHpoHejroI/v2d2mJV7ZvsdzHZxdc3K1r1fKfArMrNx0Hv5k/bVfTeeq7E6i8dvWbtIUhgMt6uizNhhz2yaB0ABAABHUhysJdUf0ul4R4YImVn0XVDVM2s13J8pS0bjacb9HgBVVWbcxs9aWzeBqmLg6fE9c8FsvHzNSKFxkPr3z9ymux76bUfTsceKbjOzyQFQSenI/QMAAJfSHOwl973e62iWCEnxy2+KzCz6rpyZnU1O2B3U3jQXB3N9LzOumE7X3pNZ1V7vVZo/AGo6Gs1eatJXvaAU6Oh9lrW1H7FkZtZFKU+jAVDZ/sgGAADtysqZQb9eR5rLsAlm0XeBl7JpxvE6s32OZrsZMGVt/TLj6nVmn7ZtrSTpzENXdfx8zYw4zsyOljOzlBkDAIDGBr00TzeSgWa9ZSNT/PK6Rpkx+s4vB7Pp6pnt9+dCNz25VvWX0KkOZg9evUjSg1q3dLTzJ2yiIjProme2EPfMMgAKAAA01u85KN14y0Vbtf/KBS3d96Kj1+veX+/Sh771QJ/3Kj3IzKLvjDFh6WtKMrOrF43oqlP312kHr+zr81T3zLbFVvZH/NNzji5/XQw8Tda4cNDPq5Ku15kNfE+lwNPuKTKzAACgsWEuoz3vyLXasnZxS/cdK/p604VHtP0czc4B05zZJZiFE4FvUpOZNcboT886UBuXj/f1eZKlIsnPkHoZ16RZazWbuNspB84F3kXfKw99avXxuuV6nVkp/EDfXZWZzfM6awAAoLZUB2sm+XX4zSeuPnFAezN8CGbhROCZ8iAihKp7Ztv5oA2nGdd+P0tVZcblx29r79pTWWbs5i/GWDHQrho9s2n+gwUAAPonjecIY6X5XaFb1i7WohG6RSWCWThS8L25dWaJaSVJflfrzKru+xiXGcfTjl1kK0cdD4CSpPGSrz01pxkDAADMiU+56p17DbPxYu25JGkeatVLBLNwIvDNXDCr4e5dcKW6Zbad98Ta+sXDRd+TtZqXCe/nZ17l0jxujBYD7SKYBQAATcQX9rtZScKFvzr/sHnbkpnZ5LliOy9luF91dwhm4UTB91KzNI8r1ePV2+lttZJm65RtF4Pw1zrum3Xxro84HgAlhVcq97A0DwAAaKLgh+dGLz5t/wHvSfvqZmYHsC/DiGJrOFHwPU3RM1shOc243QuFYc9sbaUomJ2cnpVKc1cj+5kNH63zQdtPY8VAv3h8j5snAwAAqeV7RvfdeO6gd6OpWomNsWIyMzunl2XGQ56wbojMLJwIvPRMM3almzJjydbthS0GYWBZPQSqnx9UYwPomR0r+to9LzNruVIJAAAyY7xUe8jmMK+d6xLBLJwIfE9TlBlX6KZvo9Y049tfeZq+8arTymXGk+UyYwdL81T0zLr5dB0v+dpNzywAAMiw1YtHyl9XBrDtn2/9+3Und79DQ4ZgFk4UqwZAoX4w20roaTV/SvHqxSNauXBkLpidcRfoDaJndrQQEMwCAIDMqD6327l1b61cOBfM1ist/oPD9274uPGPbVqxoPbtKa5rI5iFE4HvaXrWXaYwDSp6ZmXausBmrcpL71Qr+lUDoBy83aXAmytjdvR5OF7ytWtyuu77AAAAkGb77jVe97ZkXHvi5r0qeoK/9LJT+7lbQ4VgFk4EnqkoM05zo3mvdPMe2MQlgYVVi2mXqsqMe/F8zRhjyqXGLgdAWTsXtAMAAGRJXG1XS6PTrQ3Lx3q/M0OKYBZOhEvzEHQk+VWd+6cfvEoblo3pyh2b5t33/CPXVnwfZmalZeNFff+1Z1XcVt0z68qo41LjeOjUrom5IVDWcqEEAACkU3WtWVxtV0svz3fSfO5EMAsnAt9omqV5KvhVnxzLxov60stP1eZVC+fd940XHF7xvZU0a2tP7p3rmY3LjMP3vd/9EPHyPK4+D+Nglr5ZAACQRclleaq1M0i0l8v4DBuCWThR8D3nmcJhl/xgafYZU337t372WBjM1vi5+CqeyzJjyX1mdjwqryaYBQAAWXDi/ntVfP+ULavr3je74Wl7CGbhRIHM7DzVZcaNlGcrmXCy3T997T7d8l8P1LzSFmdmXQ6AkhKZWUdX/+Ln2zVvrVkAAID02XevcX3mJSdJkjavXKCl40VJtdeUfeszjuzZ86Y5cVs/dw30UODN9cwyfDbUzmLXcYBoJL3loq06btNyvebWO2v2UtTrme3355Tr5XnGo9Kb3RNkZgEAQDZMRwNTC4lzvM++dIfuuO+xivsduX6J0/0aVgSzcCLwmWZcrZ1ehzjwfd5Jm2SM0TOO2aBj9l2mR56YnHff6mnGrq4dlKcZO3q+uZ7ZqgFQFN4AAICUmoqSPwV/7nxm/5ULtf/KypkqWe6DbQfBLJwoeF75lxOhtsqMjalYP0yS9luxQPutmH/fcpnxTHXPbJ8HQA1omjE9swAAICvitrygwSRjqbfJgzQnAuiZhROFgJ7Zau1kZttR8sMgb7KqZ7bfH1NjRbfrzMYDoOiZBQAAWVErM1sLidkQwSycCMjMzuP16bevumfWOio0HikOJjO7h8wsAADIiKkaPbO1UGYcIpiFEwXflBvaEepXZrbuAChHS/O4KlWJ117bxQAoAACQEfHA1KCdSaFdSnNcTDALJwLf0/Qsmdkkv0+fHL5n5HtGkzNhkOdsaZ6C2zJj3zMqBV7lACjVXnsXAAAgDWaitrxS4LbiLa0YAAUnCl44zdhaV0Wvw89LXHHrdQBW9L0amdn+RnnjpcB5IDleChgABQAAMuPUg1bq0uM36uonb3b2nGnOAxDMwom47j8eApXmqWm90s/qkWLgOV+a5+lHrdOGZWNO15sdLfgMgAIAAJlR8D29bueWQe9GalBmDCfi8eL0zc5pZ2medhUDT5OOB26tWFjSuYevcfqc4yVfu+mZBQAAyCUys3AiHi8+Rd9sWb8GQElhmfHEVPRe2+z2kY4WA+2eIpgFAAB433OP1aO7J9v+uTSfJxLMwol4IhuZ2TmtBLP/9Jyjy4OV2lEqeJqYcVtmPAjjRV+7JxIDoGy6+z4AAAA6deLmvQa9C84RzMKJuMyYtWbn+BUDoGqHYKccuLKjx64eAJXVAG+sGOix3XsGvRsAAAAYgL71zBpjXmOMedAY853ov3MSt73SGHOPMeZHxpizEtvPjrbdY4x5Rb/2De4VE8GsdbVWzJDr5wCoUnIAVIbf7rGirz0MgAIAAOhCetMe/c7Mvtla+7fJDcaYQyRdLOlQSXtL+pwx5oDo5rdLOkPSA5LuMMbcaq39YZ/3EQ4EfmWZcZpr83uln0vlJKcZ9/u5Bmm85GsXS/MAAADk0iDKjHdK+oC1dkLST40x90g6JrrtHmvtvZJkjPlAdF+C2QwoTzNmAJQTxcDT76fintnspmbHioH2EMwCAIAcevGT99eOA1YMejcGqt9L81xljPmeMeZdxpil0ba1ku5P3OeBaFu97ciAQlRTO8UAqJp6nTfNT89suM5sXLpuld0sNAAAQNK1Zx6o7fss6/px0nzq1FVm1hjzOUmra9z0KknvlPR6heeXr5f0JkmXd/N8iee9QtIVkrRhw4ZePCT6rMAAqLZ84uoTu/r5YlXPbJo/pBoZKwayVvr91KxGi+1PfQYAAEB6dRXMWmtPb+V+xph/kPSJ6NsHJa1P3Lwu2qYG26uf9yZJN0nS9u3bSfWlQNwzS2a2NVvWLu7q54uBr8k8LM1TCgPY3ZPTBLMAAAA5089pxmsS354v6QfR17dKutgYUzLG7Ctps6RvSLpD0mZjzL7GmKLCIVG39mv/4FacmZ0mM+vE/DLjbKZm4zV4d9M3CwAAMM+9f3VO0/uk+SyxnwOg/sYYs1VhYug+SVdKkrX2TmPMBxUOdpqW9CJr7YwkGWOuknSbJF/Su6y1d/Zx/+BQEPXMTs9meRzR8CgGniamwwAvy0vzjJfCj7BdLM8DAAAwj9fPtSCHQN+CWWvtHze47QZJN9TY/ilJn+rXPmFw4mnGk2Rma+p1T2sp8DSRyMym+pJbA2PFysxslgN3AAAAVOr3NGNAUlj2Ks2tM4v+KiUHQGU4Fz5WDK/H7Z6gzBgAAKATaV4JgmAWTsQDoOiZdaMYeJqcmQ2XrLGZTcyWM7OUGQMAAOQPwSycKMTTjGezmyUcJkXfk7Vhj7KU5aV5wmB2DwOgAAAAcodgFk4EHtOMG+n1tOFiEPUoT89muMiYAVAAAADdSnPOg2AWTsytM0sw60IymJWyuzRPeQBUomc2q1loAAAAVCKYhRPxAKipGStr091ongblYDbum82o8gCoeJpxpvPQAAAASCKYhROBT5mxS/HFg3JmNqPXDnzPqBR42k2ZMQAAQEfSfJ5IMAsnytOMGQDlRJyZnZiezfzaq+OlgJ5ZAACADqX5XDEY9A4gHwreXJkx5uv1FbFS1QCoFF9wa2q04JfLjAEAACBdd8YB+ulvdg16N/qOYBZOMADKrbnMbBjkZblHebzkMwAKAAAg4erTNg96F5ygzBhOBF5UZkww60QpCKf8TuagzHismCgzzvhrBQAAwByCWThhjFHBN5qaZd6sC8lpxlK2y4zHir72UGYMAADQkTSfmxPMwpnA88qZ2SwHV53o9fuRnGac9csHYWaWYBYAAKATaV7GkWAWzgS+YQCUI8WgcmmeLF89GC/5LM0DAACQQwSzcKbgewyAciRZZmxtpmNZjRUrpxmbTL9aAAAAxAhm4UzBN5omM+tEXGY8MZ39iwdjxUC7J8LMLEcXAABAe9J8/kQwC2cCz9PUbPaDq2FQqiozzvLSPGNFX7unZlLd7wEAAID2EczCGTKzDfQ41kz2zGY9yBsrBrJW+v0UF0oAAADaleZTRYJZOBPEPbPWKsOJwqEwb2meDL/f46VwTd1dDIECAADoQHqjWYJZOBMOgErvL0ualHtmp2ZT/PHUmtFCGMzungiHQGU5cAcAAMAcglk4U/CNpumZdSLwPXlGmpyJArwB708/jZcCSdLuqenMl1QDAABgDsEsnAk8embr6cdyMqXAj3pmsz8ASpJ2Tcw0uScAAACqpTkXQDALZwLWmXWqGHhhMJvxQuOxYpiZ3TNJMAsAAJAnBLNwpuAbglmHioE3NwBqwPvST+XMLAOgAAAA2pbmtAfBLJwp+J6mZ7OeJxweRd/TRFRmnGXlntkomM1y4A4AANBraT5XJJiFM4E3N82YgKP/SlGZsZTtCb9xZnb35AwXSgAAAHKEYBbOFHyjacqMa+pHsDnXM5tt5WCWAVAAAAC5QjALZ4KozBhuJHtms5wLjwdA0TMLAADQvjQ3ARLMwpmCZ8plr+i/ou8lluYZ9N70j+8ZlQKPacYAAABNPPuEfQa9Cz1FMAtnwgFQBLOuFMs9s+m92taq8VJQzsxmeU1dAACAbrzmqYfO28YAKKAFgW80PZPi35Y+6kf4VQzCacb9evxhMlrwtXtiJtUfxgAAAIOQ5vMnglk4U/A9Tc3EZa9ZD68GL1lmnHXjJV+7KTMGAADIFYJZOBN4hgFQDpUKfnkAVNavHYwVAwZAAQAA5AzBLJwJosws3MhTZnasSGYWAACgE0wzBlpQ9I2mZmyqf2F6zetjxrSyZzbbqdmxYlAOZrP9SgEAABAjmIUzgR8ebiRn53hR/W8/eohLgafJ6RlZ2cyXGYc9s9NcJgEAAGhTmqv4CGbhTOCHEdU00WyZ18fUbDHwNDmTnzLjXROUGQMAAOQJwSycKXjh4Ubf7By/jynTuGdWyn7p7Vgx0B4GQAEAALTlKVtW68DVCwe9Gx0LBr0DyI84Mzs1YzMfXLWq3z2zs1aaycEE6fGir91TM5rNQxoaAACgB1YtKumdzzpq0LvRFTKzcKbgk5mtVu6Z7cNjF4Pw/Z6Yns38ur6jxUDWShNTM9lPQwMAAEASwSwcKsQ9sznIFLaqnzFm0Y+D2ez3ko6XfEmibxYAAKBFWShoI5iFMwE9s/P4UZ1xP8pjk5nZrBsrhh0Tu+mbBQAAyA2CWTgz1zOb/eCqVXPBbO8fu1RRZtz7xx8mY8UoMztJZhYAACAvCGbhTFz2Oj2TgZqGHol7WS2Z2a7EweweglkAAIDcIJiFM0EczNIzWxZPM+5rZnZqJgeZ2bDMeNfkNPOfAAAAcoJgFs7EZcZSfwcfpUk8zXimj5nZyRyUdceZ2d0MgAIAAMgNglk4U/A43KrFwexsH1KzRT8M8CamZmUynq8cL4WZ2TwE7gAAAAgRXcCZZGYWoQ3LxiT1571JZmazngmPM7MAAADIj2DQO4D8KPhcO6n2jku26Ws/eURrFo/2/LGLiZ7ZrAd7WX99AAAAmI/oAs4UyMzOs3S8qHMPX9OXx46nR09Mz2a8yHhuAJQ0NyEaAAAA2UYwC2cCemadytMAKN8z5enNAAAAaC4L64tw9gdnKjOzZM/6LQ7urM1HtjIeAgUAAIB8IJiFMwE9s04Vc5appG8WAAAgX/J1touBomfWrWTZbR7eeYJZAACAfCGYhTNMM3arIjObg2g2HgKVg5cKAAAAEczCocAjzHCpmLOLB+MlMrMAAAB5kq+zXQwUPbNuBb6n+PpBHi4jjBYYAAUAANAqm4FxxkQXcIaeWffyNASKzCwAAEBz5x62ZtC70DNdnekaY/7IGHOnMWbWGLO96rZXGmPuMcb8yBhzVmL72dG2e4wxr0hs39cY8/Vo+78aY4rd7BuGT7JnNgcrxQyFuNQ4D0vzMAAKAACguevPPmjQu9Az3aZtfiDpaZK+lNxojDlE0sWSDpV0tqR3GGN8Y4wv6e2SniLpEEnPiO4rSW+U9GZr7f6SHpP03C73DUOGnln3ikF+ArzyACgOMwAAgFzoKpi11t5lrf1RjZt2SvqAtXbCWvtTSfdIOib67x5r7b3W2klJH5C004RpoydLuiX6+fdIOq+bfcPwMcYQ0DoWL8+Th3d9nMwsAABArvSroW6tpPsT3z8Qbau3fbmkx62101XbkTEBfbNOxT2zechWjhYZAAUAANBMls4Lm579GWM+J2l1jZteZa39WO93qTljzBWSrpCkDRs2DGIX0KGC5+n3mh30buRGnpbnYQAUAABAvjQNZq21p3fwuA9KWp/4fl20TXW2PyJpiTEmiLKzyfvX2qebJN0kSdu3b8/AUOn8KASeNDHovciPcmY2B4XGY2RmAQAA2pD+MKpfaZtbJV1sjCkZY/aVtFnSNyTdIWlzNLm4qHBI1K3WWivpC5KeHv38ZZIGkvVFf8U9s9kPrYZDKUdL88TTjPMQuAMAAKD7pXnON8Y8IOl4SZ80xtwmSdbaOyV9UNIPJX1G0oustTNR1vUqSbdJukvSB6P7StL1kq41xtyjsIf2H7vZNwynQo7KXodBnnpmWZoHAACgHek/QeyqLs9a+xFJH6lz2w2Sbqix/VOSPlVj+70Kpx0jwxgA5VYxR5nZ8RJlxgAAAK2jzBhoC5lZt/I0AGq0QGYWAACgmSxV7OXnTBdDgXVm3ZorM87++05mFgAAIF8IZuEUmVm3clVmHA+Ayn7cDgAA0DWb/ipjglm4Rc+sW6Xy0jzZN8oAKAAAgKayVLFHMAunCl5+pusOgzz1zLLOLAAAQOuycD6enzNdDIVCkIHfmhTJ09I8vmc0UuAjDQAAIC8484NTgcch51KeglmJ7CwAAECr6JkF2lSgZ9apUpCvPtIx+mYBAAAaytLZOMEsnCIz61Y5M5upj636CGYBAADyg8gCTjHN2K08DYCSKDMGAADIk3yd6WLg4uAqL5nCQctbz+x4icwsAABAI1k6LySNAafIzLoVB7N5sWJBSZPTs4PeDQAAADhAMAungpyVvQ5aqdwzmw9/+YeHanKGYBYAAKCZDAwzJpiFWwUvL2HVcCj3zGapnqSBpePFQe8CAADAUMtSux9pMjhFZtatvJUZAwAAID8404VTBYJZp4o5KzMGAABAfhBZwKkCA6CcytvSPAAAAGgsS91nnOnCqcDL11Ixg5a3pXkAAACQHwSzcIqledwqBay7CgAAgGwimIVTlBm7Rc8sAAAAsopgFk4xAMqt8jqz1BkDAABA2UpyEFnAKZbmcYuleQAAAJBVnOnCqYKXpWtBwy+eZsy7DgAAgCRr7aB3oWsEs3AqzsxS9eoGmVkAAABUyNB5OGe6cIoBUG6xNA8AAACyimAWTjEAyq3AMwSyAAAAyCQiCzgV0DPrlDFGRd+TyVI9CQAAADqWpfNCglk4RWbWvWLgZao3AgAAAJAIZuFYQM+scyWGQAEAAKBK+mcZE8zCMTKz7pUCn8QsAAAAJGVrMCiRBZyKpxlnqVZ/2LE8DwAAALKIs1w4FXgccq4VfS9TV+AAAADQvSycHhJZwCl6Zt0jMwsAAIBq9MwCbaJn1r3Rgk9GHAAAAJKykZGNBYPeAeQLwax7rzjnIHnUGQMAACBjCGbhVOARVLm2bcPSQe8CAAAAhozNQJ0xaTI4RWYWAAAAGByToYo9Igs4VR4AlZ3fIQAAACB1shDTEszCqQKDiAAAAICBo8wYaFMhyMAlIAAAACClsnQ2TjALp1giBgAAAEAvEFnAqYKfpWtBAAAAQDrZDNQZE8zCKWOMfJbnAQAAAAYiC4OfYgSzcK7gm0zV6gMAAABpk4Uleghm4RwTjQEAAAB0i6gCzgX0zQIAAAADRc8s0IHA57ADAAAABiFLDX9EFXCuwAAoAAAAYCCs0p+RjRHMwrlCwGEHAAAAoDvBoHcA+XPKASu038oFg94NAAAAIHeyVGZMMAvnXrtzy6B3AQAAAEDKUe8JAAAAADmThc5ZglkAAAAAyIvsVBkTzAIAAABAbmQhJRshmAUAAAAApA7BLAAAAADkBWXGAAAAAAAMDsEsAAAAACB1ugpmjTF/ZIy50xgza4zZnti+jzFmjzHmO9F//ztx21HGmO8bY+4xxrzNGGOi7cuMMf9mjLk7+v/SbvYNAAAAAJBd3WZmfyDpaZK+VOO2n1hrt0b/vSCx/Z2Sni9pc/Tf2dH2V0j6vLV2s6TPR98DAAAAADBPV8GstfYua+2PWr2/MWaNpEXW2tuttVbSeyWdF928U9J7oq/fk9gOAAAAAECFfvbM7muM+bYx5ovGmJOibWslPZC4zwPRNklaZa19KPr6l5JW9XHfAAAAACC/MrDebNDsDsaYz0laXeOmV1lrP1bnxx6StMFa+4gx5ihJHzXGHNrqTllrrTGm7ttrjLlC0hWStGHDhlYfFgAAAAByzWRoaZ6mway19vR2H9RaOyFpIvr6v4wxP5F0gKQHJa1L3HVdtE2SHjbGrLHWPhSVI/+qwePfJOkmSdq+fXsGrikAAAAAANrRlzJjY8wKY4wffb1J4aCne6My4t8aY46LphhfKinO7t4q6bLo68sS2wEAAAAAqNDt0jznG2MekHS8pE8aY26Lbtoh6XvGmO9IukXSC6y1j0a3vVDSzZLukfQTSZ+Ott8o6QxjzN2STo++BwAAAABgnqZlxo1Yaz8i6SM1tn9I0ofq/Mw3JW2psf0RSad1sz8AAAAAgHzo5zRjAAAAAAD6gmAWAAAAAHImC1N0CWYBAAAAAKlDMAsAAAAAOZOF5WYJZgEAAAAgZygzBgAAAACkRhYysjGCWQAAAABA6hDMAgAAAEDOWJv+QmOCWQAAAABA6hDMAgAAAEDOGJP+7lmCWQAAAADIGcqMAQAAAACpkYWMbIxgFgAAAACQOgSzAAAAAIDUCQa9AwAAAAAAN8aLvi7avl4XHr1+0LvSNYJZAAAAAMgJY4ze+PTDB70bPUGZMQAAAAAgdQhmAQAAAACpQzALAAAAAEgdglkAAAAAQOoQzAIAAAAAUodgFgAAAACQOgSzAAAAAIDUIZgFAAAAAKQOwSwAAAAAIHUIZgEAAAAAqUMwCwAAAABIHYJZAAAAAEDqEMwCAAAAAFKHYBYAAAAAkDoEswAAAACA1CGYBQAAAACkDsEsAAAAACB1CGYBAAAAAKljrLWD3oeuGGN+Lelng94P5M5ekn4z6J0AGuAYxbDjGMWw4xjFsMvTMbrRWruiemPqg1lgEIwx37TWbh/0fgD1cIxi2HGMYthxjGLYcYxSZgwAAAAASCGCWQAAAABA6hDMAp25adA7ADTBMYphxzGKYccximGX+2OUnlkAAAAAQOqQmQUAAAAApA7BLCDJGLPeGPMFY8wPjTF3GmOuibYvM8b8mzHm7uj/S6PtxhjzNmPMPcaY7xljtlU93iJjzAPGmL8fxOtB9vTyGDXG/E30GHdF9zGDel3Ijg6O0YOMMf9pjJkwxvxps8cButWrYzS6bYkx5hZjzH9Hn6XHD+I1IXs6OE4vif7Of98Y8zVjzBGJxzrbGPOj6FzgFYN6Tf1EMAuEpiVdZ609RNJxkl5kjDlE0iskfWGfKc0AAAPVSURBVN5au1nS56PvJekpkjZH/10h6Z1Vj/d6SV9ysePIjZ4co8aYEyQ9SdLhkrZIOlrSyQ5fB7Kr3WP0UUkvlvS3LT4O0K1eHaOS9FZJn7HWHiTpCEl39XvnkRvtHqc/lXSytfYwheefN0mSMcaX9HaF5wOHSHpGFj9LCWYBSdbah6y134q+/p3CP0prJe2U9J7obu+RdF709U5J77Wh2yUtMcaskSRjzFGSVkn6rMOXgIzr4TFqJY1IKkoqSSpIetjZC0FmtXuMWmt/Za29Q9JUi48DdKVXx6gxZrGkHZL+MbrfpLX2cScvApnXwXH6NWvtY9H22yWti74+RtI91tp7rbWTkj4QPUamEMwCVYwx+0g6UtLXJa2y1j4U3fRLhUGqFH6o3J/4sQckrTXGeJLeJKmiHAnopW6OUWvtf0r6gqSHov9us9aSUUBPtXiMtvs4QM90eYzuK+nXkt5tjPm2MeZmY8x4v/YV+dXBcfpcSZ+Ovq55HtCXHR0gglkgwRizQNKHJL3EWvvb5G02HP3dbPz3CyV9ylr7QJ92ETnX7TFqjNlf0sEKr9yulfRkY8xJfdpd5FAPPkebPg7QjR4co4GkbZLeaa09UtIuzZV8Aj3R7nFqjDlVYTB7vbOdHAIEs0DEGFNQ+KHxfmvth6PNDyfKh9dI+lW0/UFJ6xM/vi7adrykq4wx9ynssbnUGHOjg91HDvToGD1f0u3W2iestU8ovILL4BL0RJvHaLuPA3StR8foA5IesNbGFQO3KAxugZ5o9zg1xhwu6WZJO621j0Sb650HZArBLKBw8qvC3pe7rLV/l7jpVkmXRV9fJuljie2XRhNjj5P0P1GPwyXW2g3W2n0Ulhq/11rL1Vp0rVfHqKSfSzrZGBNEfyxPFoNL0AMdHKPtPg7QlV4do9baX0q63xhzYLTpNEk/7PHuIqfaPU6NMRskfVjSH1trf5y4/x2SNhtj9jXGFCVdHD1GppgwSw3kmzHmRElflvR9SbPR5j9T2KPwQUkbJP1M0oXW2kejD5q/l3S2pN2SnmOt/WbVYz5b0nZr7VVOXgQyrVfHaDTd8B0Kh5dYhdM4r3X6YpBJHRyjqyV9U9Ki6P5PKJy4eXitx7HWfsrRS0FG9eoYtdb+1hizVWEmrCjpXoWfsY8J6FIHx+nNki6ItknStLV2e/RY50h6iyRf0rustTc4eyGOEMwCAAAAAFKHMmMAAAAAQOoQzAIAAAAAUodgFgAAAACQOgSzAAAAAIDUIZgFAAAAAKQOwSwAAAAAIHUIZgEAAAAAqUMwCwAAAABInf8PczkeEpO0VPAAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 1152x576 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AiDa4AToWnzp",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 500
        },
        "outputId": "29860e39-4723-4b97-f56e-e13d5fad8d41"
      },
      "source": [
        "plt.figure(figsize=(16,8))\n",
        "plt.plot(df_new['<CLOSE>'], label='Close Price history')"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7efd95300210>]"
            ]
          },
          "metadata": {},
          "execution_count": 14
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1152x576 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1vZkESASbesL"
      },
      "source": [
        "## **Calculating the technical indicators**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iqjLW__ObesM",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "688b3a50-e91f-4af9-a655-ddb5cd2cbb48"
      },
      "source": [
        "#1. Simple n day moving average\n",
        "#A moving average (MA) is a widely used technical indicator that smooths out price trends by filtering out the “noise” from random short-term price fluctuations.\n",
        "##The most common applications of moving averages are to identify trend direction and to determine support and resistance levels.\n",
        "def moving_average(df, n):\n",
        "    \"\"\"Calculate the moving average for the given data.\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    MA = pd.Series(df['<CLOSE>'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))\n",
        "    df = df.join(MA)\n",
        "    return df\n",
        "df_new = moving_average(df_new,10)\n",
        "df_new.tail()"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-9e1816cc-9b9f-469e-84af-24316460c654\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9e1816cc-9b9f-469e-84af-24316460c654')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9e1816cc-9b9f-469e-84af-24316460c654 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9e1816cc-9b9f-469e-84af-24316460c654');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>    <LOW>  ...    <VOL>  change_in_price    MA_10\n",
              "datetime                               ...                                   \n",
              "2020-05-26  18000.0  18417.0  17960.0  ...  2316851           -813.0  17926.7\n",
              "2020-05-27  17186.0  18990.0  17186.0  ...  6415054           -164.0  17761.4\n",
              "2020-05-30  18824.0  18824.0  18401.0  ...  6822804            895.0  17849.8\n",
              "2020-05-31  17882.0  19690.0  17882.0  ...  3692994            264.0  18004.2\n",
              "2020-06-01  19980.0  19997.0  19096.0  ...  3209736            305.0  18264.2\n",
              "\n",
              "[5 rows x 8 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "080HB4bObesP",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "03323689-96aa-4950-81d2-438fb76863b0"
      },
      "source": [
        "#2. Weighted Moving Average (WMA)\n",
        "def weighted_moving_average(df, n):\n",
        "    \"\"\"\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    WMA = pd.Series(df['<CLOSE>'].ewm(span=n, min_periods=n).mean(), name='WMA_' + str(n))\n",
        "    df = df.join(WMA)\n",
        "    return df\n",
        "df_new = weighted_moving_average(df_new,10)\n",
        "df_new.tail()"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-dee73a9b-de29-44da-be8a-c3a4d7958e7b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "      <td>18259.663642</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "      <td>18199.361162</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "      <td>18312.750041</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "      <td>18453.522761</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "      <td>18624.154986</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-dee73a9b-de29-44da-be8a-c3a4d7958e7b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-dee73a9b-de29-44da-be8a-c3a4d7958e7b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-dee73a9b-de29-44da-be8a-c3a4d7958e7b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>    <LOW>  ...  change_in_price    MA_10        WMA_10\n",
              "datetime                               ...                                        \n",
              "2020-05-26  18000.0  18417.0  17960.0  ...           -813.0  17926.7  18259.663642\n",
              "2020-05-27  17186.0  18990.0  17186.0  ...           -164.0  17761.4  18199.361162\n",
              "2020-05-30  18824.0  18824.0  18401.0  ...            895.0  17849.8  18312.750041\n",
              "2020-05-31  17882.0  19690.0  17882.0  ...            264.0  18004.2  18453.522761\n",
              "2020-06-01  19980.0  19997.0  19096.0  ...            305.0  18264.2  18624.154986\n",
              "\n",
              "[5 rows x 9 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lTZOxNVrbesS",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "6ee2cdca-2337-4873-a402-a3a130d303fc"
      },
      "source": [
        "#3. Momentum\n",
        "def momentum(df, n):\n",
        "    \"\"\"\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    M = pd.Series(df['<CLOSE>'].diff(n), name='MOM_' + str(n))\n",
        "    df = df.join(M)\n",
        "    return df\n",
        "df_new = momentum(df_new,10)\n",
        "df_new.tail()"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-314b7c1c-5f87-4d7f-babc-e78729584d61\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "      <td>18259.663642</td>\n",
              "      <td>-2014.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "      <td>18199.361162</td>\n",
              "      <td>-1653.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "      <td>18312.750041</td>\n",
              "      <td>884.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "      <td>18453.522761</td>\n",
              "      <td>1544.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "      <td>18624.154986</td>\n",
              "      <td>2600.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-314b7c1c-5f87-4d7f-babc-e78729584d61')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-314b7c1c-5f87-4d7f-babc-e78729584d61 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-314b7c1c-5f87-4d7f-babc-e78729584d61');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>    <LOW>  ...    MA_10        WMA_10  MOM_10\n",
              "datetime                               ...                               \n",
              "2020-05-26  18000.0  18417.0  17960.0  ...  17926.7  18259.663642 -2014.0\n",
              "2020-05-27  17186.0  18990.0  17186.0  ...  17761.4  18199.361162 -1653.0\n",
              "2020-05-30  18824.0  18824.0  18401.0  ...  17849.8  18312.750041   884.0\n",
              "2020-05-31  17882.0  19690.0  17882.0  ...  18004.2  18453.522761  1544.0\n",
              "2020-06-01  19980.0  19997.0  19096.0  ...  18264.2  18624.154986  2600.0\n",
              "\n",
              "[5 rows x 10 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9WasGdiqbesV",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "7767a08c-6b70-44f0-a494-22f209d40a89"
      },
      "source": [
        "#4. Stochastic K%\n",
        "def stochastic_oscillator_k(df):\n",
        "    \"\"\"Calculate stochastic oscillator %K for given data.\n",
        "    :param df: pandas.DataFrame\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    SOk = pd.Series((df['<CLOSE>'] - df['<LOW>']) / (df['<HIGH>'] - df['<LOW>']), name='SO_k')\n",
        "    df = df.join(SOk)\n",
        "    return df\n",
        "df_new = stochastic_oscillator_k(df_new)\n",
        "df_new.tail()"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-fc56ad2c-60e3-464b-8178-527d23772551\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_k</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "      <td>18259.663642</td>\n",
              "      <td>-2014.0</td>\n",
              "      <td>0.288840</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "      <td>18199.361162</td>\n",
              "      <td>-1653.0</td>\n",
              "      <td>0.411308</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "      <td>18312.750041</td>\n",
              "      <td>884.0</td>\n",
              "      <td>0.997636</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "      <td>18453.522761</td>\n",
              "      <td>1544.0</td>\n",
              "      <td>0.666482</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "      <td>18624.154986</td>\n",
              "      <td>2600.0</td>\n",
              "      <td>0.328524</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fc56ad2c-60e3-464b-8178-527d23772551')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-fc56ad2c-60e3-464b-8178-527d23772551 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-fc56ad2c-60e3-464b-8178-527d23772551');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>    <LOW>  ...        WMA_10  MOM_10      SO_k\n",
              "datetime                               ...                                \n",
              "2020-05-26  18000.0  18417.0  17960.0  ...  18259.663642 -2014.0  0.288840\n",
              "2020-05-27  17186.0  18990.0  17186.0  ...  18199.361162 -1653.0  0.411308\n",
              "2020-05-30  18824.0  18824.0  18401.0  ...  18312.750041   884.0  0.997636\n",
              "2020-05-31  17882.0  19690.0  17882.0  ...  18453.522761  1544.0  0.666482\n",
              "2020-06-01  19980.0  19997.0  19096.0  ...  18624.154986  2600.0  0.328524\n",
              "\n",
              "[5 rows x 11 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "d5u31Lq2besX",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "21fb9661-81f5-4a26-8ff7-fd8cfa889d55"
      },
      "source": [
        "#5. Stochastic D%\n",
        "def stochastic_oscillator_d(df, n):\n",
        "    \"\"\"Calculate stochastic oscillator %D for given data.\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    SOk = pd.Series((df['<CLOSE>'] - df['<LOW>']) / (df['<HIGH>'] - df['<LOW>']), name='SO%k')\n",
        "    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO_' + str(n))\n",
        "    df = df.join(SOd)\n",
        "    return df\n",
        "df_new = stochastic_oscillator_d(df_new,10)\n",
        "df_new.tail()"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-f5f41c70-4403-4636-ad60-26a49280d542\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_k</th>\n",
              "      <th>SO_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "      <td>18259.663642</td>\n",
              "      <td>-2014.0</td>\n",
              "      <td>0.288840</td>\n",
              "      <td>0.539398</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "      <td>18199.361162</td>\n",
              "      <td>-1653.0</td>\n",
              "      <td>0.411308</td>\n",
              "      <td>0.516099</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "      <td>18312.750041</td>\n",
              "      <td>884.0</td>\n",
              "      <td>0.997636</td>\n",
              "      <td>0.603680</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "      <td>18453.522761</td>\n",
              "      <td>1544.0</td>\n",
              "      <td>0.666482</td>\n",
              "      <td>0.615102</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "      <td>18624.154986</td>\n",
              "      <td>2600.0</td>\n",
              "      <td>0.328524</td>\n",
              "      <td>0.562985</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f5f41c70-4403-4636-ad60-26a49280d542')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f5f41c70-4403-4636-ad60-26a49280d542 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f5f41c70-4403-4636-ad60-26a49280d542');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>    <LOW>  ...  MOM_10      SO_k     SO_10\n",
              "datetime                               ...                            \n",
              "2020-05-26  18000.0  18417.0  17960.0  ... -2014.0  0.288840  0.539398\n",
              "2020-05-27  17186.0  18990.0  17186.0  ... -1653.0  0.411308  0.516099\n",
              "2020-05-30  18824.0  18824.0  18401.0  ...   884.0  0.997636  0.603680\n",
              "2020-05-31  17882.0  19690.0  17882.0  ...  1544.0  0.666482  0.615102\n",
              "2020-06-01  19980.0  19997.0  19096.0  ...  2600.0  0.328524  0.562985\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oki_RUaObesb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 140
        },
        "outputId": "fcec1d04-96de-44bb-9b9f-d2f004616eb3"
      },
      "source": [
        "#6. Relative Strength Index\n",
        "#Error\n",
        "\"\"\"\n",
        "def relative_strength_index(df, n):\n",
        "    Calculate Relative Strength Index(RSI) for given data.\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    i = df.index[0]\n",
        "    UpI = [0]\n",
        "    DoI = [0]\n",
        "    while i + 1 <= df.index[-1]:\n",
        "        UpMove = float(df.loc[i + 1, 'high']) - float(df.loc[i, 'high'])\n",
        "        DoMove = float(df.loc[i, 'low']) - float(df.loc[i + 1, 'low'])\n",
        "        if UpMove > DoMove and UpMove > 0:\n",
        "            UpD = UpMove\n",
        "        else:\n",
        "            UpD = 0\n",
        "        UpI.append(UpD)\n",
        "        if DoMove > UpMove and DoMove > 0:\n",
        "            DoD = DoMove\n",
        "        else:\n",
        "            DoD = 0\n",
        "        DoI.append(DoD)\n",
        "        i = i + 1\n",
        "    UpI = pd.Series(UpI)\n",
        "\n",
        "    DoI = pd.Series(DoI)\n",
        "    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())\n",
        "    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())\n",
        "\n",
        "    # rsi = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))\n",
        "    rsi = pd.DataFrame(PosDI / (PosDI + NegDI), columns=['RSI_' + str(n)])\n",
        "    rsi = rsi.set_index(df.index)\n",
        "    df = df.join(rsi)\n",
        "    return df\n",
        "\"\"\""
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "\"\\ndef relative_strength_index(df, n):\\n    Calculate Relative Strength Index(RSI) for given data.\\n    :param df: pandas.DataFrame\\n    :param n:\\n    :return: pandas.DataFrame\\n    i = df.index[0]\\n    UpI = [0]\\n    DoI = [0]\\n    while i + 1 <= df.index[-1]:\\n        UpMove = float(df.loc[i + 1, 'high']) - float(df.loc[i, 'high'])\\n        DoMove = float(df.loc[i, 'low']) - float(df.loc[i + 1, 'low'])\\n        if UpMove > DoMove and UpMove > 0:\\n            UpD = UpMove\\n        else:\\n            UpD = 0\\n        UpI.append(UpD)\\n        if DoMove > UpMove and DoMove > 0:\\n            DoD = DoMove\\n        else:\\n            DoD = 0\\n        DoI.append(DoD)\\n        i = i + 1\\n    UpI = pd.Series(UpI)\\n\\n    DoI = pd.Series(DoI)\\n    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())\\n    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())\\n\\n    # rsi = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))\\n    rsi = pd.DataFrame(PosDI / (PosDI + NegDI), columns=['RSI_' + str(n)])\\n    rsi = rsi.set_index(df.index)\\n    df = df.join(rsi)\\n    return df\\n\""
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SzBt7Xdobese",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 388
        },
        "outputId": "ef9d681c-3c5b-4a48-81cc-a24e271665aa"
      },
      "source": [
        "#7. Signal\n",
        "def macd(df, n_fast, n_slow):\n",
        "    \"\"\"Calculate MACD, MACD Signal and MACD difference\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n_fast:\n",
        "    :param n_slow:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    EMAfast = pd.Series(df['<CLOSE>'].ewm(span=n_fast, min_periods=n_slow).mean())\n",
        "    EMAslow = pd.Series(df['<CLOSE>'].ewm(span=n_slow, min_periods=n_slow).mean())\n",
        "    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))\n",
        "    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))\n",
        "    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))\n",
        "    df = df.join(MACD)\n",
        "    df = df.join(MACDsign)\n",
        "    df = df.join(MACDdiff)\n",
        "    return df\n",
        "df_new = macd(df_new,12,26)\n",
        "df_new.head()"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-2661f341-fca2-45cc-bfd0-9abc11bf007a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_k</th>\n",
              "      <th>SO_10</th>\n",
              "      <th>MACD_12_26</th>\n",
              "      <th>MACDsign_12_26</th>\n",
              "      <th>MACDdiff_12_26</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2001-03-25</th>\n",
              "      <td>2140.0</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>2139.0</td>\n",
              "      <td>2140.0</td>\n",
              "      <td>349714320</td>\n",
              "      <td>163488</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-26</th>\n",
              "      <td>2135.0</td>\n",
              "      <td>2136.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>37030936</td>\n",
              "      <td>17577</td>\n",
              "      <td>-40.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-27</th>\n",
              "      <td>2100.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2045.0</td>\n",
              "      <td>2050.0</td>\n",
              "      <td>200173239</td>\n",
              "      <td>97608</td>\n",
              "      <td>-50.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.090909</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-28</th>\n",
              "      <td>2049.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2020.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>120265895</td>\n",
              "      <td>59019</td>\n",
              "      <td>50.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2001-03-31</th>\n",
              "      <td>2101.0</td>\n",
              "      <td>2205.0</td>\n",
              "      <td>2100.0</td>\n",
              "      <td>2205.0</td>\n",
              "      <td>187171518</td>\n",
              "      <td>85296</td>\n",
              "      <td>105.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2661f341-fca2-45cc-bfd0-9abc11bf007a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2661f341-fca2-45cc-bfd0-9abc11bf007a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2661f341-fca2-45cc-bfd0-9abc11bf007a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>  <HIGH>   <LOW>  ...  MACD_12_26  MACDsign_12_26  MACDdiff_12_26\n",
              "datetime                             ...                                            \n",
              "2001-03-25   2140.0  2140.0  2139.0  ...         NaN             NaN             NaN\n",
              "2001-03-26   2135.0  2136.0  2100.0  ...         NaN             NaN             NaN\n",
              "2001-03-27   2100.0  2100.0  2045.0  ...         NaN             NaN             NaN\n",
              "2001-03-28   2049.0  2100.0  2020.0  ...         NaN             NaN             NaN\n",
              "2001-03-31   2101.0  2205.0  2100.0  ...         NaN             NaN             NaN\n",
              "\n",
              "[5 rows x 15 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cfobRA3dbesh"
      },
      "source": [
        "#8. Larry Williams R%\n",
        "#lEFT"
      ],
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tAAH6tYWbesm",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 388
        },
        "outputId": "0fe64caa-921c-4b59-f829-9f54e45ef030"
      },
      "source": [
        "#9. Accumulation / Distribution\n",
        "def accumulation_distribution(df, n):\n",
        "    \"\"\"Calculate Accumulation/Distribution for given data.\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    ad = (2 * df['<CLOSE>'] - df['<HIGH>'] - df['<LOW>']) / (df['<HIGH>'] - df['<LOW>']) * df['<VOL>']\n",
        "    M = ad.diff(n - 1)\n",
        "    N = ad.shift(n - 1)\n",
        "    ROC = M / N\n",
        "    AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))\n",
        "    df = df.join(AD)\n",
        "    return df\n",
        "df_new = accumulation_distribution(df_new,10)\n",
        "df_new.tail()"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-6ac7ede4-590a-43af-a3df-2c42f91385e7\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_k</th>\n",
              "      <th>SO_10</th>\n",
              "      <th>MACD_12_26</th>\n",
              "      <th>MACDsign_12_26</th>\n",
              "      <th>MACDdiff_12_26</th>\n",
              "      <th>Acc/Dist_ROC_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "      <td>18259.663642</td>\n",
              "      <td>-2014.0</td>\n",
              "      <td>0.288840</td>\n",
              "      <td>0.539398</td>\n",
              "      <td>617.206061</td>\n",
              "      <td>951.060144</td>\n",
              "      <td>-333.854083</td>\n",
              "      <td>-0.652602</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "      <td>18199.361162</td>\n",
              "      <td>-1653.0</td>\n",
              "      <td>0.411308</td>\n",
              "      <td>0.516099</td>\n",
              "      <td>544.399044</td>\n",
              "      <td>869.727924</td>\n",
              "      <td>-325.328879</td>\n",
              "      <td>0.202279</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "      <td>18312.750041</td>\n",
              "      <td>884.0</td>\n",
              "      <td>0.997636</td>\n",
              "      <td>0.603680</td>\n",
              "      <td>552.548544</td>\n",
              "      <td>806.292048</td>\n",
              "      <td>-253.743504</td>\n",
              "      <td>-2.330912</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "      <td>18453.522761</td>\n",
              "      <td>1544.0</td>\n",
              "      <td>0.666482</td>\n",
              "      <td>0.615102</td>\n",
              "      <td>573.696459</td>\n",
              "      <td>759.772930</td>\n",
              "      <td>-186.076471</td>\n",
              "      <td>-1.665997</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "      <td>18624.154986</td>\n",
              "      <td>2600.0</td>\n",
              "      <td>0.328524</td>\n",
              "      <td>0.562985</td>\n",
              "      <td>608.057971</td>\n",
              "      <td>729.429938</td>\n",
              "      <td>-121.371967</td>\n",
              "      <td>-0.668024</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6ac7ede4-590a-43af-a3df-2c42f91385e7')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-6ac7ede4-590a-43af-a3df-2c42f91385e7 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-6ac7ede4-590a-43af-a3df-2c42f91385e7');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>  ...  MACDdiff_12_26  Acc/Dist_ROC_10\n",
              "datetime                      ...                                 \n",
              "2020-05-26  18000.0  18417.0  ...     -333.854083        -0.652602\n",
              "2020-05-27  17186.0  18990.0  ...     -325.328879         0.202279\n",
              "2020-05-30  18824.0  18824.0  ...     -253.743504        -2.330912\n",
              "2020-05-31  17882.0  19690.0  ...     -186.076471        -1.665997\n",
              "2020-06-01  19980.0  19997.0  ...     -121.371967        -0.668024\n",
              "\n",
              "[5 rows x 16 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XCFLfxtUbesq",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 388
        },
        "outputId": "bfb292dc-c5e7-490c-f3c4-22e115f2def2"
      },
      "source": [
        "#10. Commodity Channel Index\n",
        "def commodity_channel_index(df, n):\n",
        "    \"\"\"Calculate Commodity Channel Index for given data.\n",
        "    :param df: pandas.DataFrame\n",
        "    :param n:\n",
        "    :return: pandas.DataFrame\n",
        "    \"\"\"\n",
        "    PP = (df['<HIGH>'] + df['<LOW>'] + df['<CLOSE>']) / 3\n",
        "    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),\n",
        "                    name='CCI_' + str(n))\n",
        "    df = df.join(CCI)\n",
        "    return df\n",
        "df_new = commodity_channel_index(df_new,10)\n",
        "df_new.tail()"
      ],
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-6d101a50-7f9f-4202-9392-28396ee2001c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_k</th>\n",
              "      <th>SO_10</th>\n",
              "      <th>MACD_12_26</th>\n",
              "      <th>MACDsign_12_26</th>\n",
              "      <th>MACDdiff_12_26</th>\n",
              "      <th>Acc/Dist_ROC_10</th>\n",
              "      <th>CCI_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>18000.0</td>\n",
              "      <td>18417.0</td>\n",
              "      <td>17960.0</td>\n",
              "      <td>18092.0</td>\n",
              "      <td>41636609960</td>\n",
              "      <td>2316851</td>\n",
              "      <td>-813.0</td>\n",
              "      <td>17926.7</td>\n",
              "      <td>18259.663642</td>\n",
              "      <td>-2014.0</td>\n",
              "      <td>0.288840</td>\n",
              "      <td>0.539398</td>\n",
              "      <td>617.206061</td>\n",
              "      <td>951.060144</td>\n",
              "      <td>-333.854083</td>\n",
              "      <td>-0.652602</td>\n",
              "      <td>0.177397</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>17186.0</td>\n",
              "      <td>18990.0</td>\n",
              "      <td>17186.0</td>\n",
              "      <td>17928.0</td>\n",
              "      <td>115006593572</td>\n",
              "      <td>6415054</td>\n",
              "      <td>-164.0</td>\n",
              "      <td>17761.4</td>\n",
              "      <td>18199.361162</td>\n",
              "      <td>-1653.0</td>\n",
              "      <td>0.411308</td>\n",
              "      <td>0.516099</td>\n",
              "      <td>544.399044</td>\n",
              "      <td>869.727924</td>\n",
              "      <td>-325.328879</td>\n",
              "      <td>0.202279</td>\n",
              "      <td>0.309628</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>18824.0</td>\n",
              "      <td>18824.0</td>\n",
              "      <td>18401.0</td>\n",
              "      <td>18823.0</td>\n",
              "      <td>128426052100</td>\n",
              "      <td>6822804</td>\n",
              "      <td>895.0</td>\n",
              "      <td>17849.8</td>\n",
              "      <td>18312.750041</td>\n",
              "      <td>884.0</td>\n",
              "      <td>0.997636</td>\n",
              "      <td>0.603680</td>\n",
              "      <td>552.548544</td>\n",
              "      <td>806.292048</td>\n",
              "      <td>-253.743504</td>\n",
              "      <td>-2.330912</td>\n",
              "      <td>1.020630</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>17882.0</td>\n",
              "      <td>19690.0</td>\n",
              "      <td>17882.0</td>\n",
              "      <td>19087.0</td>\n",
              "      <td>70489687776</td>\n",
              "      <td>3692994</td>\n",
              "      <td>264.0</td>\n",
              "      <td>18004.2</td>\n",
              "      <td>18453.522761</td>\n",
              "      <td>1544.0</td>\n",
              "      <td>0.666482</td>\n",
              "      <td>0.615102</td>\n",
              "      <td>573.696459</td>\n",
              "      <td>759.772930</td>\n",
              "      <td>-186.076471</td>\n",
              "      <td>-1.665997</td>\n",
              "      <td>1.056782</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>19980.0</td>\n",
              "      <td>19997.0</td>\n",
              "      <td>19096.0</td>\n",
              "      <td>19392.0</td>\n",
              "      <td>62242497958</td>\n",
              "      <td>3209736</td>\n",
              "      <td>305.0</td>\n",
              "      <td>18264.2</td>\n",
              "      <td>18624.154986</td>\n",
              "      <td>2600.0</td>\n",
              "      <td>0.328524</td>\n",
              "      <td>0.562985</td>\n",
              "      <td>608.057971</td>\n",
              "      <td>729.429938</td>\n",
              "      <td>-121.371967</td>\n",
              "      <td>-0.668024</td>\n",
              "      <td>1.396910</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6d101a50-7f9f-4202-9392-28396ee2001c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-6d101a50-7f9f-4202-9392-28396ee2001c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-6d101a50-7f9f-4202-9392-28396ee2001c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "            <FIRST>   <HIGH>  ...  Acc/Dist_ROC_10    CCI_10\n",
              "datetime                      ...                           \n",
              "2020-05-26  18000.0  18417.0  ...        -0.652602  0.177397\n",
              "2020-05-27  17186.0  18990.0  ...         0.202279  0.309628\n",
              "2020-05-30  18824.0  18824.0  ...        -2.330912  1.020630\n",
              "2020-05-31  17882.0  19690.0  ...        -1.665997  1.056782\n",
              "2020-06-01  19980.0  19997.0  ...        -0.668024  1.396910\n",
              "\n",
              "[5 rows x 17 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HHQkL2Fybesu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "04d50620-9b92-4da1-811d-5ee364c04217"
      },
      "source": [
        "df_new.mean()"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<FIRST>            2.314985e+03\n",
              "<HIGH>             2.354751e+03\n",
              "<LOW>              2.270215e+03\n",
              "<CLOSE>            2.315439e+03\n",
              "<VALUE>            8.081839e+09\n",
              "<VOL>              2.544001e+06\n",
              "change_in_price    4.222222e+00\n",
              "MA_10              2.297528e+03\n",
              "WMA_10             2.297644e+03\n",
              "MOM_10             3.948173e+01\n",
              "SO_k                        NaN\n",
              "SO_10              5.351231e-01\n",
              "MACD_12_26         2.610116e+01\n",
              "MACDsign_12_26     2.537952e+01\n",
              "MACDdiff_12_26     6.755629e-01\n",
              "Acc/Dist_ROC_10             NaN\n",
              "CCI_10            -6.896271e-02\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mSwQUUeKbesx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fe481c27-4537-451f-8025-1b7f730ea1b9"
      },
      "source": [
        "df_new.min()"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<FIRST>             491.000000\n",
              "<HIGH>              491.000000\n",
              "<LOW>               491.000000\n",
              "<CLOSE>             506.000000\n",
              "<VALUE>            5710.000000\n",
              "<VOL>                10.000000\n",
              "change_in_price   -1642.000000\n",
              "MA_10               515.100000\n",
              "WMA_10              518.418291\n",
              "MOM_10            -4300.000000\n",
              "SO_k                      -inf\n",
              "SO_10                -4.641678\n",
              "MACD_12_26         -833.183394\n",
              "MACDsign_12_26     -759.151369\n",
              "MACDdiff_12_26     -663.921320\n",
              "Acc/Dist_ROC_10           -inf\n",
              "CCI_10               -2.846050\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mMVVklQPbes1",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "34fee77b-4b41-42ad-e2a8-a4a698cb439a"
      },
      "source": [
        "df_new.max()"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<FIRST>            2.200000e+04\n",
              "<HIGH>             2.250000e+04\n",
              "<LOW>              2.148200e+04\n",
              "<CLOSE>            2.148300e+04\n",
              "<VALUE>            3.370985e+11\n",
              "<VOL>              8.097697e+07\n",
              "change_in_price    1.311000e+03\n",
              "MA_10              2.005040e+04\n",
              "WMA_10             1.964120e+04\n",
              "MOM_10             5.604000e+03\n",
              "SO_k                        inf\n",
              "SO_10              7.132721e+00\n",
              "MACD_12_26         2.305244e+03\n",
              "MACDsign_12_26     2.060763e+03\n",
              "MACDdiff_12_26     4.236789e+02\n",
              "Acc/Dist_ROC_10             inf\n",
              "CCI_10             2.839801e+00\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uNTpjP7Xbes4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "54f90399-80f4-4280-a19f-2078c9b998c6"
      },
      "source": [
        "df_new.std()"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<FIRST>            2.370610e+03\n",
              "<HIGH>             2.423969e+03\n",
              "<LOW>              2.305411e+03\n",
              "<CLOSE>            2.366677e+03\n",
              "<VALUE>            2.214442e+10\n",
              "<VOL>              4.476661e+06\n",
              "change_in_price    1.131251e+02\n",
              "MA_10              2.297954e+03\n",
              "WMA_10             2.293439e+03\n",
              "MOM_10             4.988782e+02\n",
              "SO_k                        NaN\n",
              "SO_10              3.134283e-01\n",
              "MACD_12_26         2.073677e+02\n",
              "MACDsign_12_26     1.981625e+02\n",
              "MACDdiff_12_26     5.159681e+01\n",
              "Acc/Dist_ROC_10             NaN\n",
              "CCI_10             1.261867e+00\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vfZGT9zFbes9"
      },
      "source": [
        "## **Normalizing Data**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8tYyjDyjbes-",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 388
        },
        "outputId": "68668eb5-6219-4d52-ba93-3de0621d6b6d"
      },
      "source": [
        "data = (df_new - df_new.mean()) / (df_new.max() - df_new.min())\n",
        "data.tail()"
      ],
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-ccb6ad1e-e71b-41af-b8e4-893d7486fc8b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>&lt;FIRST&gt;</th>\n",
              "      <th>&lt;HIGH&gt;</th>\n",
              "      <th>&lt;LOW&gt;</th>\n",
              "      <th>&lt;CLOSE&gt;</th>\n",
              "      <th>&lt;VALUE&gt;</th>\n",
              "      <th>&lt;VOL&gt;</th>\n",
              "      <th>change_in_price</th>\n",
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_k</th>\n",
              "      <th>SO_10</th>\n",
              "      <th>MACD_12_26</th>\n",
              "      <th>MACDsign_12_26</th>\n",
              "      <th>MACDdiff_12_26</th>\n",
              "      <th>Acc/Dist_ROC_10</th>\n",
              "      <th>CCI_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>0.729230</td>\n",
              "      <td>0.729804</td>\n",
              "      <td>0.747453</td>\n",
              "      <td>0.752089</td>\n",
              "      <td>0.099540</td>\n",
              "      <td>-0.002805</td>\n",
              "      <td>-0.276743</td>\n",
              "      <td>0.800048</td>\n",
              "      <td>0.834712</td>\n",
              "      <td>-0.207339</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000363</td>\n",
              "      <td>0.188344</td>\n",
              "      <td>0.328266</td>\n",
              "      <td>-0.307585</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.043329</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>0.691386</td>\n",
              "      <td>0.755838</td>\n",
              "      <td>0.710580</td>\n",
              "      <td>0.744270</td>\n",
              "      <td>0.317191</td>\n",
              "      <td>0.047804</td>\n",
              "      <td>-0.056967</td>\n",
              "      <td>0.791586</td>\n",
              "      <td>0.831559</td>\n",
              "      <td>-0.170889</td>\n",
              "      <td>NaN</td>\n",
              "      <td>-0.001616</td>\n",
              "      <td>0.165146</td>\n",
              "      <td>0.299423</td>\n",
              "      <td>-0.299747</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.066585</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>0.767540</td>\n",
              "      <td>0.748296</td>\n",
              "      <td>0.768462</td>\n",
              "      <td>0.786936</td>\n",
              "      <td>0.357000</td>\n",
              "      <td>0.052840</td>\n",
              "      <td>0.301652</td>\n",
              "      <td>0.796111</td>\n",
              "      <td>0.837488</td>\n",
              "      <td>0.085270</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.005823</td>\n",
              "      <td>0.167742</td>\n",
              "      <td>0.276928</td>\n",
              "      <td>-0.233927</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.191632</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>0.723744</td>\n",
              "      <td>0.787644</td>\n",
              "      <td>0.743737</td>\n",
              "      <td>0.799521</td>\n",
              "      <td>0.185132</td>\n",
              "      <td>0.014189</td>\n",
              "      <td>0.087971</td>\n",
              "      <td>0.804015</td>\n",
              "      <td>0.844850</td>\n",
              "      <td>0.151910</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.006793</td>\n",
              "      <td>0.174481</td>\n",
              "      <td>0.260431</td>\n",
              "      <td>-0.171710</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.197991</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>0.821285</td>\n",
              "      <td>0.801592</td>\n",
              "      <td>0.801571</td>\n",
              "      <td>0.814061</td>\n",
              "      <td>0.160667</td>\n",
              "      <td>0.008221</td>\n",
              "      <td>0.101855</td>\n",
              "      <td>0.817324</td>\n",
              "      <td>0.853773</td>\n",
              "      <td>0.258534</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.002366</td>\n",
              "      <td>0.185429</td>\n",
              "      <td>0.249671</td>\n",
              "      <td>-0.112217</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.257811</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ccb6ad1e-e71b-41af-b8e4-893d7486fc8b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ccb6ad1e-e71b-41af-b8e4-893d7486fc8b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ccb6ad1e-e71b-41af-b8e4-893d7486fc8b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "             <FIRST>    <HIGH>  ...  Acc/Dist_ROC_10    CCI_10\n",
              "datetime                        ...                           \n",
              "2020-05-26  0.729230  0.729804  ...              NaN  0.043329\n",
              "2020-05-27  0.691386  0.755838  ...              NaN  0.066585\n",
              "2020-05-30  0.767540  0.748296  ...              NaN  0.191632\n",
              "2020-05-31  0.723744  0.787644  ...              NaN  0.197991\n",
              "2020-06-01  0.821285  0.801592  ...              NaN  0.257811\n",
              "\n",
              "[5 rows x 17 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YcwrDXOzbetB",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 563
        },
        "outputId": "9ac093c3-d801-4c44-8b65-3502a5218ba2"
      },
      "source": [
        "import seaborn as sns\n",
        "plt.figure(1 , figsize = (17 , 8))\n",
        "cor = sns.heatmap(data.corr(), annot = True)"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1224x576 with 2 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Nw7aDwgebetG",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "47c5b05a-5e1a-4f72-f47d-6172176ffa77"
      },
      "source": [
        "#To check null values\n",
        "data.isnull().sum()"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<FIRST>               0\n",
              "<HIGH>                0\n",
              "<LOW>                 0\n",
              "<CLOSE>               0\n",
              "<VALUE>               0\n",
              "<VOL>                 0\n",
              "change_in_price       1\n",
              "MA_10                 9\n",
              "WMA_10                9\n",
              "MOM_10               10\n",
              "SO_k               4087\n",
              "SO_10                 9\n",
              "MACD_12_26           25\n",
              "MACDsign_12_26       33\n",
              "MACDdiff_12_26       33\n",
              "Acc/Dist_ROC_10    4087\n",
              "CCI_10                9\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QN8FItuAbetJ"
      },
      "source": [
        "#Removing the null values\n",
        "data = data.fillna(data.mean())"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "shbMSgnWbetN"
      },
      "source": [
        "data = data.dropna(how=\"all\",axis=1)"
      ],
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hKHZhcd8betR",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        },
        "outputId": "7bb31d2a-9d73-4f20-8814-5c054f35f5af"
      },
      "source": [
        "##Defining the features and labels\n",
        "X = data.iloc[:,7:]\n",
        "X.tail()"
      ],
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-7d2692c9-efd7-4915-9537-10da4de5b2ed\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>MA_10</th>\n",
              "      <th>WMA_10</th>\n",
              "      <th>MOM_10</th>\n",
              "      <th>SO_10</th>\n",
              "      <th>MACD_12_26</th>\n",
              "      <th>MACDsign_12_26</th>\n",
              "      <th>MACDdiff_12_26</th>\n",
              "      <th>CCI_10</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>datetime</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2020-05-26</th>\n",
              "      <td>0.800048</td>\n",
              "      <td>0.834712</td>\n",
              "      <td>-0.207339</td>\n",
              "      <td>0.000363</td>\n",
              "      <td>0.188344</td>\n",
              "      <td>0.328266</td>\n",
              "      <td>-0.307585</td>\n",
              "      <td>0.043329</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-27</th>\n",
              "      <td>0.791586</td>\n",
              "      <td>0.831559</td>\n",
              "      <td>-0.170889</td>\n",
              "      <td>-0.001616</td>\n",
              "      <td>0.165146</td>\n",
              "      <td>0.299423</td>\n",
              "      <td>-0.299747</td>\n",
              "      <td>0.066585</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-30</th>\n",
              "      <td>0.796111</td>\n",
              "      <td>0.837488</td>\n",
              "      <td>0.085270</td>\n",
              "      <td>0.005823</td>\n",
              "      <td>0.167742</td>\n",
              "      <td>0.276928</td>\n",
              "      <td>-0.233927</td>\n",
              "      <td>0.191632</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-05-31</th>\n",
              "      <td>0.804015</td>\n",
              "      <td>0.844850</td>\n",
              "      <td>0.151910</td>\n",
              "      <td>0.006793</td>\n",
              "      <td>0.174481</td>\n",
              "      <td>0.260431</td>\n",
              "      <td>-0.171710</td>\n",
              "      <td>0.197991</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2020-06-01</th>\n",
              "      <td>0.817324</td>\n",
              "      <td>0.853773</td>\n",
              "      <td>0.258534</td>\n",
              "      <td>0.002366</td>\n",
              "      <td>0.185429</td>\n",
              "      <td>0.249671</td>\n",
              "      <td>-0.112217</td>\n",
              "      <td>0.257811</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7d2692c9-efd7-4915-9537-10da4de5b2ed')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
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
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-7d2692c9-efd7-4915-9537-10da4de5b2ed button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-7d2692c9-efd7-4915-9537-10da4de5b2ed');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "               MA_10    WMA_10  ...  MACDdiff_12_26    CCI_10\n",
              "datetime                        ...                          \n",
              "2020-05-26  0.800048  0.834712  ...       -0.307585  0.043329\n",
              "2020-05-27  0.791586  0.831559  ...       -0.299747  0.066585\n",
              "2020-05-30  0.796111  0.837488  ...       -0.233927  0.191632\n",
              "2020-05-31  0.804015  0.844850  ...       -0.171710  0.197991\n",
              "2020-06-01  0.817324  0.853773  ...       -0.112217  0.257811\n",
              "\n",
              "[5 rows x 8 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-RGklbtnbetV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "15ade6c7-db4a-49c2-a06b-e32319abd5e7"
      },
      "source": [
        "Y = data.iloc[:,3]\n",
        "Y.tail()"
      ],
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "datetime\n",
              "2020-05-26    0.752089\n",
              "2020-05-27    0.744270\n",
              "2020-05-30    0.786936\n",
              "2020-05-31    0.799521\n",
              "2020-06-01    0.814061\n",
              "Name: <CLOSE>, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1Tn80CwmbetX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "473ca209-8666-4a5d-ecc4-f4ea84ab435e"
      },
      "source": [
        "X.isnull().sum()"
      ],
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MA_10             0\n",
              "WMA_10            0\n",
              "MOM_10            0\n",
              "SO_10             0\n",
              "MACD_12_26        0\n",
              "MACDsign_12_26    0\n",
              "MACDdiff_12_26    0\n",
              "CCI_10            0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IRkULxA7sCFy"
      },
      "source": [
        "## Visualising the technical indicators i.e. our features"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0UP3raQjr7Ty",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "bc47308a-088b-423b-b795-32b5e3bba3f6"
      },
      "source": [
        "X.hist(bins=50,figsize=(20,15))"
      ],
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7efd94f45750>,\n",
              "        <matplotlib.axes._subplots.AxesSubplot object at 0x7efd94f453d0>,\n",
              "        <matplotlib.axes._subplots.AxesSubplot object at 0x7efd936b3350>],\n",
              "       [<matplotlib.axes._subplots.AxesSubplot object at 0x7efd9366a950>,\n",
              "        <matplotlib.axes._subplots.AxesSubplot object at 0x7efd93622f50>,\n",
              "        <matplotlib.axes._subplots.AxesSubplot object at 0x7efd935dee50>],\n",
              "       [<matplotlib.axes._subplots.AxesSubplot object at 0x7efd935a1410>,\n",
              "        <matplotlib.axes._subplots.AxesSubplot object at 0x7efd93558850>,\n",
              "        <matplotlib.axes._subplots.AxesSubplot object at 0x7efd93558890>]],\n",
              "      dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 37
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1440x1080 with 9 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QjK-zRQgbeta"
      },
      "source": [
        "#Splitting data into training and test set\n",
        "#from sklearn.model_selection import train_test_split\n",
        "total_days = len(X)\n",
        "hold_days=30\n",
        "x_train = X[0:(total_days-hold_days)]\n",
        "y_train = Y[0:(total_days-hold_days)]"
      ],
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bdSjkPr_betc"
      },
      "source": [
        "# Predicting using Decision Trees"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7RX-ujiabetd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b6150cd1-405d-43e1-97cc-df280d95b070"
      },
      "source": [
        "#Decision Tree\n",
        "from sklearn.tree import DecisionTreeRegressor\n",
        "Regressor = DecisionTreeRegressor(max_depth=10)\n",
        "Regressor.fit(x_train,y_train)"
      ],
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DecisionTreeRegressor(max_depth=10)"
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tmAA0_rPbetl"
      },
      "source": [
        "## Evaluating the Model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pCwX7Hwpbetm",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2f5f6e30-6506-4736-ad75-253132fe886d"
      },
      "source": [
        "y_pred = Regressor.predict(X)\n",
        "from sklearn.metrics import r2_score,mean_squared_error\n",
        "mse = mean_squared_error(Y,y_pred)\n",
        "rmse = np.sqrt(mse)\n",
        "rmse"
      ],
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.023487284455733753"
            ]
          },
          "metadata": {},
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EW6CbXYgSzMZ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "0d178200-68b2-453f-db98-6ddf16b358ad"
      },
      "source": [
        "plt.plot(X.index,Y)\n",
        "plt.plot(X.index, y_pred, c='#f5ef42')\n",
        "plt.show()"
      ],
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CiWT_doibeto"
      },
      "source": [
        "# Using Bagging Regressor"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6KWr75vGbetp"
      },
      "source": [
        "#import necessary libraries\n",
        "from sklearn.ensemble import BaggingRegressor"
      ],
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6m5OsIUsbetr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6205a342-0452-47ab-c64a-eea36fce7309"
      },
      "source": [
        "regr = BaggingRegressor(base_estimator=Regressor,\n",
        "n_estimators=50, random_state=0)\n",
        "regr.fit(x_train,y_train)\n",
        "    "
      ],
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=10),\n",
              "                 n_estimators=50, random_state=0)"
            ]
          },
          "metadata": {},
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KvdNXLTXbetw"
      },
      "source": [
        "## Evaluating Bagging Regressor"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Af_3vxHkbetx"
      },
      "source": [
        "y_pred_br = regr.predict(X)"
      ],
      "execution_count": 44,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PXoR2Bxpbetz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8955fb04-73b7-47a8-d126-e78c0bc6e9ab"
      },
      "source": [
        "from sklearn.metrics import r2_score,mean_squared_error\n",
        "mse = mean_squared_error(Y,y_pred_br)\n",
        "rmse = np.sqrt(mse)\n",
        "rmse"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.01828923172281144"
            ]
          },
          "metadata": {},
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NRz2RSH3fQGM",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "0fc6d0bc-b13c-4d8e-df3d-d99ae6db7f50"
      },
      "source": [
        "plt.plot(X.index,Y)\n",
        "plt.plot(X.index, y_pred_br, c='#f5ef42')\n",
        "plt.show()"
      ],
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "phoe90D7bet1"
      },
      "source": [
        "# Using Random Forest"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cM6Op45nbet2",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3069c116-ed92-4daf-9e9d-149709cfdedd"
      },
      "source": [
        "#importing the libraries\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "rf = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=1)\n",
        "rf.fit(x_train,y_train)"
      ],
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestRegressor(max_depth=10, n_estimators=50, random_state=1)"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j4GVbyhAbet9"
      },
      "source": [
        "## Evaluating Random Forest"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Rp1ihgYjbet-"
      },
      "source": [
        "y_pred_rf = rf.predict(X)\n"
      ],
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "J6mI-BUbbeuA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8e52a108-8834-437e-b054-0836c4c5f775"
      },
      "source": [
        "from sklearn.metrics import r2_score,mean_squared_error\n",
        "mse = mean_squared_error(Y,y_pred_rf)\n",
        "rmse = np.sqrt(mse)\n",
        "rmse"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.019200663958058854"
            ]
          },
          "metadata": {},
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uj2B69g3gnSx",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "3ba2dc00-c506-4f5d-ded1-95dbc79da7b5"
      },
      "source": [
        "plt.plot(X.index,Y)\n",
        "plt.plot(X.index, y_pred_rf, c='#f5ef42')\n",
        "plt.show()"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6e1uCe8lbeuD"
      },
      "source": [
        " ## Using AdaBoost Regressor"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uX6q6_kGbeuD",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4c9282e9-6bf0-4c0c-dce8-8b222c514f63"
      },
      "source": [
        "#importing necessary libraries\n",
        "from sklearn.ensemble import AdaBoostRegressor\n",
        "adb_reg = AdaBoostRegressor(base_estimator=Regressor,n_estimators=100,learning_rate=0.1)\n",
        "adb_reg.fit(x_train,y_train)\n",
        "\n"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10),\n",
              "                  learning_rate=0.1, n_estimators=100)"
            ]
          },
          "metadata": {},
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pTiNntHcbeuH"
      },
      "source": [
        "## Evaluating AdaBoost Regressor"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pGnz1YOibeuI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "40068056-c85f-479e-fc1b-25f413e3ac54"
      },
      "source": [
        "y_pred_adb = adb_reg.predict(X)\n",
        "from sklearn.metrics import r2_score,mean_squared_error\n",
        "mse = mean_squared_error(Y,y_pred_adb)\n",
        "rmse = np.sqrt(mse)\n",
        "rmse"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.016913788759981337"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BfFSD3AChDmn",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        },
        "outputId": "7c71443f-d53a-4861-98c8-66360e25bd0b"
      },
      "source": [
        "plt.plot(X.index,Y)\n",
        "plt.plot(X.index, y_pred_adb, c='#f5ef42')\n",
        "plt.show()"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8Q-98ibGbeuJ"
      },
      "source": [
        "## Using Gradient Boosting"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4StZOXBBbeuK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "115f501a-c703-46b4-f513-d6987f2e6880"
      },
      "source": [
        "#importing necessary libraries\n",
        "from sklearn.ensemble import GradientBoostingRegressor\n",
        "gdbt = GradientBoostingRegressor(max_depth=10,learning_rate=0.1,n_estimators=50,random_state=1)\n",
        "gdbt.fit(x_train,y_train)"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GradientBoostingRegressor(max_depth=10, n_estimators=50, random_state=1)"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fhjqy5udbeuP"
      },
      "source": [
        "## Evaluating gradient boosting"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CpcZr0-EbeuQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "dacab9a1-89c6-49d1-a05c-82acafb400b7"
      },
      "source": [
        "y_pred_gdbt = gdbt.predict(X)\n",
        "from sklearn.metrics import r2_score,mean_squared_error\n",
        "mse = mean_squared_error(Y,y_pred_gdbt)\n",
        "rmse = np.sqrt(mse)\n",
        "rmse"
      ],
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.017547477294209003"
            ]
          },
          "metadata": {},
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GdaVicrEb78c"
      },
      "source": [
        "##Using Deep Learning Models"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zNH4J9l5cAY6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 328
        },
        "outputId": "9b3bcb97-4716-4d68-efb4-7e6f6d7b886d"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "error",
          "ename": "MessageError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mMessageError\u001b[0m                              Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-56-d5df0069828e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mgoogle\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolab\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mdrive\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mdrive\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmount\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/drive'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/google/colab/drive.py\u001b[0m in \u001b[0;36mmount\u001b[0;34m(mountpoint, force_remount, timeout_ms, use_metadata_server)\u001b[0m\n\u001b[1;32m    113\u001b[0m       \u001b[0mforce_remount\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mforce_remount\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    114\u001b[0m       \u001b[0mtimeout_ms\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout_ms\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 115\u001b[0;31m       ephemeral=True)\n\u001b[0m\u001b[1;32m    116\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    117\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/google/colab/drive.py\u001b[0m in \u001b[0;36m_mount\u001b[0;34m(mountpoint, force_remount, timeout_ms, use_metadata_server, ephemeral)\u001b[0m\n\u001b[1;32m    133\u001b[0m   \u001b[0;32mif\u001b[0m \u001b[0mephemeral\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    134\u001b[0m     _message.blocking_request(\n\u001b[0;32m--> 135\u001b[0;31m         'request_auth', request={'authType': 'dfs_ephemeral'}, timeout_sec=None)\n\u001b[0m\u001b[1;32m    136\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    137\u001b[0m   \u001b[0mmountpoint\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_os\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexpanduser\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmountpoint\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/google/colab/_message.py\u001b[0m in \u001b[0;36mblocking_request\u001b[0;34m(request_type, request, timeout_sec, parent)\u001b[0m\n\u001b[1;32m    173\u001b[0m   request_id = send_request(\n\u001b[1;32m    174\u001b[0m       request_type, request, parent=parent, expect_reply=True)\n\u001b[0;32m--> 175\u001b[0;31m   \u001b[0;32mreturn\u001b[0m \u001b[0mread_reply_from_input\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrequest_id\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout_sec\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/google/colab/_message.py\u001b[0m in \u001b[0;36mread_reply_from_input\u001b[0;34m(message_id, timeout_sec)\u001b[0m\n\u001b[1;32m    104\u001b[0m         reply.get('colab_msg_id') == message_id):\n\u001b[1;32m    105\u001b[0m       \u001b[0;32mif\u001b[0m \u001b[0;34m'error'\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mreply\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 106\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mMessageError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mreply\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'error'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    107\u001b[0m       \u001b[0;32mreturn\u001b[0m \u001b[0mreply\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'data'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    108\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mMessageError\u001b[0m: Error: credential propagation was unsuccessful"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BvTCIFZ6BJgy"
      },
      "source": [
        "### Importing Libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y5R1YEo_BMOc"
      },
      "source": [
        "import tensorflow as tf \n",
        "import pandas as pd \n",
        "import numpy as np \n",
        "import matplotlib.pyplot as plt \n",
        "import keras \n",
        "from keras.models import Sequential \n",
        "from keras.layers import Dense \n",
        "from sklearn.metrics import confusion_matrix "
      ],
      "execution_count": 66,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MqGBMKv0HvVM"
      },
      "source": [
        "## Using LSTM"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "B2uIFD452b24"
      },
      "source": [
        "### Importing libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ccMkltE9ebcw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "56163003-ce18-4fc8-8d4a-2fdf432cd659"
      },
      "source": [
        "pip install nsepy"
      ],
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting nsepy\n",
            "  Downloading nsepy-0.8.tar.gz (33 kB)\n",
            "Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.7/dist-packages (from nsepy) (4.6.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from nsepy) (2.23.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from nsepy) (1.21.5)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from nsepy) (1.3.5)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from nsepy) (1.15.0)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from nsepy) (7.1.2)\n",
            "Requirement already satisfied: lxml in /usr/local/lib/python3.7/dist-packages (from nsepy) (4.2.6)\n",
            "Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->nsepy) (2018.9)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->nsepy) (2.8.2)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->nsepy) (2021.10.8)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->nsepy) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->nsepy) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->nsepy) (1.24.3)\n",
            "Building wheels for collected packages: nsepy\n",
            "  Building wheel for nsepy (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for nsepy: filename=nsepy-0.8-py3-none-any.whl size=36084 sha256=ff503173c7899b205155fc49f440c4a6d7fc99c431f996e9b6ebb605fd903b48\n",
            "  Stored in directory: /root/.cache/pip/wheels/32/ab/d9/78ceea14cdf6de83376082b3cb0c2999fd77f823e35c47b9ec\n",
            "Successfully built nsepy\n",
            "Installing collected packages: nsepy\n",
            "Successfully installed nsepy-0.8\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7fcRAPcyeVTV"
      },
      "source": [
        "from nsepy import get_history as gh\n",
        "import datetime as dt\n",
        "from matplotlib import pyplot as plt\n",
        "from sklearn import model_selection\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "from keras.layers import LSTM\n",
        "from keras.layers import Dropout"
      ],
      "execution_count": 68,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rpjHV_boZZl8"
      },
      "source": [
        "from sklearn.preprocessing import MinMaxScaler\n",
        "sc = MinMaxScaler(feature_range = (0, 1))\n",
        "X_train_scaled = sc.fit_transform(x_train)"
      ],
      "execution_count": 69,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ur_xNnK6Zela"
      },
      "source": [
        "X_train, Y_train = np.array(x_train), np.array(y_train)\n",
        "X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))"
      ],
      "execution_count": 70,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "33TKI23A2NVQ"
      },
      "source": [
        "### Building the LSTM"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SrT0kjBsc2Pm"
      },
      "source": [
        "regressor = Sequential()\n",
        "regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))\n",
        "regressor.add(Dropout(0.2))\n",
        "regressor.add(LSTM(units = 50, return_sequences = True))\n",
        "regressor.add(Dropout(0.2))\n",
        "regressor.add(LSTM(units = 50, return_sequences = True))\n",
        "regressor.add(Dropout(0.2))\n",
        "regressor.add(LSTM(units = 50))\n",
        "regressor.add(Dropout(0.2))\n",
        "regressor.add(Dense(units = 1))"
      ],
      "execution_count": 71,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L_LlaDFB2qDN"
      },
      "source": [
        "### Compiling and training the model defined in the above step.\n",
        "(Iteratively, we can increase or decrease the epochs and batch size to get more accuracy.)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NEg-dl9jeOSR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1918f700-0776-45ee-de82-d0f5e8b8baff"
      },
      "source": [
        "regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')\n",
        "regressor.fit(X_train, Y_train, epochs = 15, batch_size = 32)"
      ],
      "execution_count": 72,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/15\n",
            "127/127 [==============================] - 9s 18ms/step - loss: 0.0014\n",
            "Epoch 2/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 3.0699e-04\n",
            "Epoch 3/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 2.8150e-04\n",
            "Epoch 4/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 2.7202e-04\n",
            "Epoch 5/15\n",
            "127/127 [==============================] - 2s 19ms/step - loss: 1.7447e-04\n",
            "Epoch 6/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 2.2433e-04\n",
            "Epoch 7/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 1.9666e-04\n",
            "Epoch 8/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 1.9952e-04\n",
            "Epoch 9/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 2.0973e-04\n",
            "Epoch 10/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 2.5186e-04\n",
            "Epoch 11/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 2.4259e-04\n",
            "Epoch 12/15\n",
            "127/127 [==============================] - 2s 19ms/step - loss: 1.6096e-04\n",
            "Epoch 13/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 1.4026e-04\n",
            "Epoch 14/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 1.5357e-04\n",
            "Epoch 15/15\n",
            "127/127 [==============================] - 2s 18ms/step - loss: 1.7625e-04\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7efd235a5d90>"
            ]
          },
          "metadata": {},
          "execution_count": 72
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r2ZrIgaX03cC"
      },
      "source": [
        "X_test = np.array(X)\n",
        "X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))\n",
        "predicted_stock_price = regressor.predict(X_test)"
      ],
      "execution_count": 73,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LDrGcESafEp5"
      },
      "source": [
        "real_stock_price = np.asarray(Y)\n",
        "# real_stock_price=real_stock_price.reshape(1167,1)"
      ],
      "execution_count": 74,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "koD72IuliN0t",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a0a00de3-c72f-46a4-d381-d436ddb6ac5d"
      },
      "source": [
        "predicted_stock_price"
      ],
      "execution_count": 75,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.00582852],\n",
              "       [-0.00582852],\n",
              "       [-0.00582852],\n",
              "       ...,\n",
              "       [ 0.67274255],\n",
              "       [ 0.6856767 ],\n",
              "       [ 0.70476234]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bXqa74JLiSvC",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "62c93afd-7f51-4557-8193-508f67cab9f2"
      },
      "source": [
        "real_stock_price"
      ],
      "execution_count": 76,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-0.0083634 , -0.01027025, -0.01265381, ...,  0.78693622,\n",
              "        0.79952143,  0.81406116])"
            ]
          },
          "metadata": {},
          "execution_count": 76
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oWXVsRgp2DcB"
      },
      "source": [
        "### Plotting the results"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bTm04GWiiU9O",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 594
        },
        "outputId": "348fb212-2541-4f3f-e193-90fcc36fb903"
      },
      "source": [
        "plt.figure(figsize=(20,10))\n",
        "plt.plot(real_stock_price, color = 'green', label = 'Stock Price')\n",
        "plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Stock Price')\n",
        "plt.title('Stock Price Prediction')\n",
        "plt.xlabel('Trading Day')\n",
        "plt.ylabel('Stock Price')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "execution_count": 77,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABI8AAAJcCAYAAABwj4S5AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeZSdZZnv/e9Vu6aMQEhogQBJyxjIAIY0NKMiAZoWF3joaKuICkK7REXpdjj94tBtS59WEMXD5AgooijIQWQQwTA0NgkgraSRQZEQhEyQoYY93e8fe6hdw64UkKpdlfp+1qq1n2k/+0oVqxb55bruJ1JKSJIkSZIkSQNpanQBkiRJkiRJGr0MjyRJkiRJklSX4ZEkSZIkSZLqMjySJEmSJElSXYZHkiRJkiRJqsvwSJIkSZIkSXUZHkmSpDEvIv4YEW8ehvvuHhGbIiKzte89XCLi7og4o7z9zoi4/VXe5+cR8Z6tW50kSRqLDI8kSdKwiYjDI+L+iHg5ItZFxH0RcXD53OkRcW8DakoRsbkcCj0XERfWC4dSSn9KKU1OKRUaVcNrkVL6Xkpp8RDq+WxEXNPnvSeklL67tWuSJEljj+GRJEkaFhExFbgZ+BowDdgV+BzQ3ci6yuanlCYDxwB/D5zZ94KIaB4HNUiSJG2R4ZEkSRouewOklK5NKRVSSp0ppdtTSo9GxH7AZcCh5e6blwAiYruIuCoiVkfEMxHxzxFR/f+ViDgzIlZExMaIeCwiDur7oRGxX0T8ISLesaUCU0r/A9wDHBARs8odQe+PiD8Bv6w51ly+97SI+HZErIqI9RFxY83n/m1EPBIRL5W7reYN5Zu0pRrK935f+c+9PiJui4g9aj732Ij4n3J31yVA1Jzr1d0VEftHxB3lLrAXIuLTEXE88GlgSfln8ZvytbXjb03ln8UzEfFi+We0Xflcpeb3RMSfImJNRPzvofzZJUnS2GB4JEmShsvvgUJEfDciToiIHSonUkorgLOB/yyPhW1fPvU1YDvgL4GjgNOA9wJExKnAZ8vHpgInAWtrP7AcJt0GnJNSunZLBUbEHOAI4OGaw0cB+wHHDfCWq4GJwP7ATsBF5fscCHwLOAvYEbgcuCki2l5rDRHxVkrhzinADEpB07Xl904HfgL8MzAdeAo4rM7nTAF+AdwK7ALsCdyZUroV+DfguvLPYv4Abz+9/PVGSj+bycAlfa45HNiHUifV+eWAUJIkbQMMjyRJ0rBIKW2gFCgk4EpgdUTcFBF/MdD15TV/3g58KqW0MaX0R+DLwLvLl5wB/J+U0oOp5MmU0jM1tzgCuAk4LaV08xbKeygi1gP/D/gG8O2ac59NKW1OKXX2qW9n4ATg7JTS+pRSLqX0q/LpDwCXp5R+Xe6y+i6l8bxDtkINZwNfTCmtSCnlKQU9C8rdR38D/C6ldH1KKQd8Bfhznc/7W+DPKaUvp5S6yt/jX2/h+1TxTuDClNLTKaVNwKeAt/cZq/tcubvsN8BvgIFCKEmSNAY5Ry9JkoZNucPodICI2Be4hlLAMdBI2XSgBagNhJ6htFYSwG6UOmvqORv4VUrp7iGUdlBK6cnaAxHVaa9n67xnN2BdSmn9AOf2AN4TEefUHGul1OHzWmvYA7g4Ir5ceyml78sutdemlFJEDFb/YN+/wexC/59LM1AbBNaGVh2UupMkSdI2wM4jSZI0Ispr+3wHOKByqM8la4AcpbCkYnfgufL2s8DrB/mIs4HdI+Ki11pqnePPAtMiYvs6576QUtq+5mviUEbnhlDDs8BZfe49IaV0P/A8pVAIgCilT7sxsGcpjZxt6fMGsor+P5c88MIW3idJkrYBhkeSJGlYRMS+EfHxiJhZ3t+NUsfRA+VLXgBmRkQrQEqpAPwQ+EJETCmPZX2MUrcSlEa7zouIN0TJnrULRwMbgeOBIyPigq3950kpPQ/8HPi/EbFDRLRExJHl01cCZ0fEX5VrmxQRJ5bXGXqtLgM+FRH7Q3VR8VPL534G7B8Rp5RHyD4MvK7OfW4Gdo6Ij0ZEW/l7/Fflcy8As6JmcfI+rgXOjYjZETGZnjWS8lvhzydJkkY5wyNJkjRcNgJ/Bfw6IjZTCo1+C3y8fP6XwO+AP0fEmvKxc4DNwNPAvcD3KS1ETUrpR8AXysc2AjcC02o/MKX0EnAscEJE/Msw/JneTak76n+AF4GPlj93GXAmpUWk1wNPUh7Xe61SSjcA/w78ICI2UPoenlA+twY4FbiA0uLhewH31bnPRkrfm7dQGjF7gtIC2AA/Kr+ujYiHBnj7tygtFr4U+APQRelnJUmSxoFIaUtdypIkSZIkSRqv7DySJEmSJElSXYZHkiRJkiRJqsvwSJIkSZIkSXUZHkmSJEmSJKmu5kYX8EpNnz49zZo1q9FlSJIkSZIkbTOWL1++JqU0Y6BzYy48mjVrFsuWLWt0GZIkSZIkSduMiHim3jnH1iRJkiRJklSX4ZEkSZIkSZLqMjySJEmSJElSXWNuzaOB5HI5Vq5cSVdXV6NL0RjQ3t7OzJkzaWlpaXQpkiRJkiSNettEeLRy5UqmTJnCrFmziIhGl6NRLKXE2rVrWblyJbNnz250OZIkSZIkjXrbxNhaV1cXO+64o8GRtigi2HHHHe1SkyRJkiRpiLaJ8AgwONKQ+d+KJEmSJElDt82ER5IkSZIkSdr6DI+2ki984Qvsv//+zJs3jwULFvDrX/8agK985St0dHS8qnt+9rOf5Utf+tIWr9l1111ZsGABBxxwADfddNOA11122WVcddVVr6oOSZIkSZI0fm0TC2Y32n/+539y880389BDD9HW1saaNWvIZrNAKTx617vexcSJE4ft888991zOO+88VqxYwRFHHMGLL75IU1NPLpjP5zn77LOH7fMlSZIkSdK2y86jreD5559n+vTptLW1ATB9+nR22WUXvvrVr7Jq1Sre+MY38sY3vhGAa6+9lrlz53LAAQfwiU98onqPW2+9lYMOOoj58+dzzDHH9PuMK6+8khNOOIHOzs66dey33340NzezZs0ajj76aD760Y+ycOFCLr744l5dTE8++SRvfvObmT9/PgcddBBPPfUUAP/xH//BwQcfzLx58/jMZz6z1b4/kiRJkiRp7NrmOo8+eutHeeTPj2zVey543QK+cvxX6p5fvHgxn//859l7771585vfzJIlSzjqqKP48Ic/zIUXXshdd93F9OnTWbVqFZ/4xCdYvnw5O+ywA4sXL+bGG2/ksMMO48wzz2Tp0qXMnj2bdevW9br/JZdcwh133MGNN95YDagG8utf/5qmpiZmzJgBQDabZdmyZUBpvK3ine98J5/85Cc5+eST6erqolgscvvtt/PEE0/wX//1X6SUOOmkk1i6dClHHnnka/jOSZIkSZKksW6bC48aYfLkySxfvpx77rmHu+66iyVLlnDBBRdw+umn97ruwQcf5Oijj66GO+985ztZunQpmUyGI488ktmzZwMwbdq06nuuuuoqdtttN2688UZaWloG/PyLLrqIa665hilTpnDddddVnya2ZMmSftdu3LiR5557jpNPPhmA9vZ2AG6//XZuv/12DjzwQAA2bdrEE088YXgkSZIkSdI4t82FR4N1CA2nTCbD0UcfzdFHH83cuXP57ne/2y88ejXmzp3LI488wsqVK6vhUl+VNY/6mjRp0pA/J6XEpz71Kc4666xXXaskSZIkSdr2uObRVvD444/zxBNPVPcfeeQR9thjDwCmTJnCxo0bAVi0aBG/+tWvWLNmDYVCgWuvvZajjjqKQw45hKVLl/KHP/wBoNfY2oEHHsjll1/OSSedxKpVq15zrVOmTGHmzJnceOONAHR3d9PR0cFxxx3Ht771LTZt2gTAc889x4svvviaP0+SJEmSJI1t21znUSNs2rSJc845h5deeonm5mb23HNPrrjiCgA+8IEPcPzxx7PLLrtw1113ccEFF/DGN76RlBInnngib33rWwG44oorOOWUUygWi+y0007ccccd1fsffvjhfOlLX+LEE0/kjjvuYPr06a+p3quvvpqzzjqL888/n5aWFn70ox+xePFiVqxYwaGHHgqURvGuueYadtppp9f0WZIkSZIkaWyLlFKja3hFFi5cmCqLQFesWLGC/fbbr0EVaSzyvxlJkiRJknpExPKU0sKBzjm2JkmSJEmSpLoMjyRJkiRJklSX4ZEkSZIkSZLqMjySJEmSJElSXYZHkiRJkiRJqsvwSJIkSZIkaQv+4eZ/4IrlVzS6jIYwPNpKMpkMCxYs4IADDuDUU0+lo6PjVd/r9NNP5/rrrwfgjDPO4LHHHqt77d13383999//ij9j1qxZrFmzpt/xb33rW8ydO5d58+ZxwAEH8NOf/hSA73znO6xateoVf07lvR/60Ie2eM2MGTNYsGABc+bM4corrxzwuptuuokLLrjgVdUhSZIkSdKrddnyyzjr5rMaXUZDGB5tJRMmTOCRRx7ht7/9La2trVx22WW9zufz+Vd132984xvMmTOn7vlXGx4NZOXKlXzhC1/g3nvv5dFHH+WBBx5g3rx5wGsLj4ZqyZIlPPLII9x99918+tOf5oUXXuh1Pp/Pc9JJJ/HJT35yWOuQJEmSJEk9DI+GwRFHHMGTTz7J3XffzRFHHMFJJ53EnDlzKBQK/OM//iMHH3ww8+bN4/LLLwcgpcSHPvQh9tlnH9785jfz4osvVu919NFHs2zZMgBuvfVWDjroIObPn88xxxzDH//4Ry677DIuuugiFixYwD333MPq1at529vexsEHH8zBBx/MfffdB8DatWtZvHgx+++/P2eccQYppX51v/jii0yZMoXJkycDMHnyZGbPns3111/PsmXLeOc738mCBQvo7Ozkzjvv5MADD2Tu3Lm8733vo7u7G4AHH3yQv/7rv2b+/PksWrSIjRs39vqMn/3sZxx66KEDdj1V7LTTTrz+9a/nmWee4fTTT+fss8/mr/7qr/inf/qnXl1ML7zwAieffDLz589n/vz51RDtmmuuYdGiRSxYsICzzjqLQqHwqn6OkiRJkiQJmhtdwFb30Y/CI49s3XsuWABf+cqQLs3n8/z85z/n+OOPB+Chhx7it7/9LbNnz+aKK65gu+2248EHH6S7u5vDDjuMxYsX8/DDD/P444/z2GOP8cILLzBnzhze97739brv6tWrOfPMM1m6dCmzZ89m3bp1TJs2jbPPPpvJkydz3nnnAfD3f//3nHvuuRx++OH86U9/4rjjjmPFihV87nOf4/DDD+f888/nZz/7Gd/85jf71T5//nz+4i/+gtmzZ3PMMcdwyimn8Ja3vIX/9b/+F5dccglf+tKXWLhwIV1dXZx++unceeed7L333px22mlceumlfPCDH2TJkiVcd911HHzwwWzYsIEJEyZU73/DDTdw4YUXcsstt7DDDjvU/R4+/fTTPP300+y5555AqSPq/vvvJ5PJ8J3vfKd63Yc//GGOOuoobrjhBgqFAps2bWLFihVcd9113HfffbS0tPDBD36Q733ve5x22mlD+vlJkiRJkqTetr3wqEE6OztZsGABUOo8ev/738/999/PokWLmD17NgC33347jz76aHU9o5dffpknnniCpUuX8o53vINMJsMuu+zCm970pn73f+CBBzjyyCOr95o2bdqAdfziF7/otUbShg0b2LRpE0uXLuUnP/kJACeeeOKA4U0mk+HWW2/lwQcf5M477+Tcc89l+fLlfPazn+113eOPP87s2bPZe++9AXjPe97D17/+dY455hh23nlnDj74YACmTp1afc8vf/lLli1bxu23397reK3rrruOe++9l7a2Ni6//PLqn/HUU08lk8n0u/6Xv/wlV111VbX27bbbjquvvprly5dXa+js7GSnnXYa8PMkSZIkSdKWbXvh0RA7hLa2yppHfU2aNKm6nVLia1/7Gscdd1yva2655ZatVkexWOSBBx6g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\n",
            "text/plain": [
              "<Figure size 1440x720 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7hCjZ69viYqn",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8d10036f-b940-4e7d-e773-0ee976f38367"
      },
      "source": [
        "from sklearn.metrics import mean_squared_error as MSE\n",
        "rmse = MSE(real_stock_price,predicted_stock_price)**(1/2)\n",
        "rmse"
      ],
      "execution_count": 78,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.011414482776863672"
            ]
          },
          "metadata": {},
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "CIcDi7MKb4M7"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
