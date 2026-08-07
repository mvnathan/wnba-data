# Google Colab Loader for wnba-data

Use this exact cell in Colab:

```python
import sys
import requests

url = (
    "https://raw.githubusercontent.com/"
    "YOUR_GITHUB_USERNAME/wnba-data/main/"
    "colab/load_repository_data.py"
)

code = requests.get(url, timeout=60)
code.raise_for_status()

exec(code.text)

games, quarter_scores, team_games, metadata = (
    load_wnba_repository(
        github_user="YOUR_GITHUB_USERNAME",
        repository="wnba-data",
        branch="main",
        force_refresh=False,
    )
)
```

This loader downloads compact Parquet files directly from GitHub and converts date columns to timezone-aware UTC.
