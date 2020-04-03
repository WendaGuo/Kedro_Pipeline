import sys
from datetime import datetime, timedelta

from airflow import DAG
from slugify import slugify

from kedro_airflow.runner import AirflowRunner
from kedro.context import load_context  # isort:skip


# Get our project source onto the python path
sys.path.append("/Users/wen/Desktop/AD/boston_house_prices/src")

# Path to Kedro project directory
project_path = "/Users/wen/Desktop/AD/boston_house_prices"


# Default arguments for all the Airflow operators
default_args = {
    "owner": "kedro",
    "start_date": datetime(2015, 6, 1),
    "depends_on_past": True,
    "wait_for_downstream": True,
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Arguments for specific Airflow operators, keyed by Kedro node name
def operator_specific_arguments(task_id):
    return {}


# Injest Airflow's context, may modify the data catalog as necessary
# also a good place to init Spark
def process_context(data_catalog, **airflow_context):
    # you could put all the airflow context into the catalog as a new dataset
    for key in ["dag", "conf", "macros", "task", "task_instance", "ti", "var"]:
        del airflow_context[key]  # drop unpicklable things
    data_catalog.add_feed_dict({"airflow_context": airflow_context}, replace=True)

    # or add just the ones you need into Kedro parameters
    parameters = data_catalog.load("parameters")
    parameters["airflow_ds"] = airflow_context["ds"]
    data_catalog.load("parameters", parameters)

    return data_catalog


# Construct a DAG and then call into Kedro to have the operators constructed
dag = DAG(
    slugify("BostonHousePrices"),
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)


_context = load_context(project_path)
data_catalog = _context.catalog
pipeline = _context.pipeline

runner = AirflowRunner(
    dag=dag,
    process_context=process_context,
    operator_arguments=operator_specific_arguments,
)

runner.run(pipeline, data_catalog)
