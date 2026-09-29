# Divvy Bike Analysis

Author: Farhan Sadeek (solo project, no collaborators). Original repo: https://github.com/SadeekFarhan21/Divvy_Bike_Analysis (live: https://divvy.farhansadeek.com). Code license: MIT per the original README (the original repo has no LICENSE file). Trip data is Divvy's, owned by the City of Chicago and used under https://divvybikes.com/data-license-agreement; the raw CSVs are not included here and are not in the original repo either.

Note: this copy omits the rendered site, the About Me page, logos and stale files. The analysis is descriptive EDA only. `index.qmd` cannot be rendered without downloading the 12 monthly 2024 Divvy CSVs into `datasets/`. Its prose observations disagree with its own charts in places; see `results/` for numbers read from the charts.

---


This repository contains an analysis of Divvy bike share data. The analysis includes data cleaning, exploratory data analysis, and visualization to uncover insights about bike usage patterns.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to analyze Divvy bike share data to understand usage patterns, identify trends, and provide actionable insights. The analysis covers various aspects such as ride types, user categories, ride lengths, and popular stations.

## Data Sources
The data used in this project is sourced from the Divvy bike share system. The dataset includes information about individual rides, such as start and end times, start and end stations, rideable types, and user types.

## Installation
To run the analysis, you need to have R and RStudio installed on your machine. Additionally, you need to install the required R packages.

```r
install.packages(c("tidyverse", "ggplot2", "dplyr", "lubridate"))
```

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/SadeekFarhan21/Divvy_Bike_Analysis.git
    ```
2. Open the project in RStudio.
3. Run the analysis scripts to generate the visualizations and insights.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

This project is licensed under the MIT License.