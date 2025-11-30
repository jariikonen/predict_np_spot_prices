# predict_np_spot_prices

This is a miniproject for the Introduction to Data Science course at the University of Helsinki.
The goal of the project was to analyze the formation of Finnish electricity spot prices on the Nordpool electricity exchange and see, whether it could be possible to create forecasts that could help consumers plan their electricity usage for a longer period than just the next day. 

## Usage

### Sync uv and activate virtual environment

You need to have [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) installed. After that
run
```
uv sync
```
and navigate to the project directory. Activate the virtual environment by sourcing `.venv/bin/activate`:

```
. .venv/bin/activate
```

### CLI commands

There is a command line application that allows fetching, updating, displaying and preprocessing data from ENTSO-E.
Call the script (`pnsp`) like this:

```
uv run pnsp --help
```

```
usage: predict-np-spot-prices [-h] {fetch,update,show,preprocess,run_app} ...

positional arguments:
  {fetch,update,show,preprocess,run_app}
                        Command to execute (fetch, update, show, preprocess, run_app, weather).
    fetch               Fetch data from ENTSO-E.
    update              Update data to the current day from ENTSO-E.
    show                Show downloaded data items.
    preprocess          Preprocesses the data in the data directory.
    run_app             Runs the Streamlit app.

options:
  -h, --help            show this help message and exit
```

To fetch data from ENTSO-E, you need to place your API key in the `secrets.toml` file in the .streamlit directory at the project root.

### Streamlit application

The streamlit application can be run like this:

```
uv run pnsp run_app
```

### Sources of data

The project contains data from [ENTSO-E](https://www.entsoe.eu/data/transparency-platform/) and [Finnish Meteorological Institute](https://en.ilmatieteenlaitos.fi/open-data-licence).
