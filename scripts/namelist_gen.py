
import sys, yaml
from jinja2 import Environment, FileSystemLoader

# Read arguments from command line
aqm_date = sys.argv[1]
print(aqm_date)

# Set up jinja2 environment
env = Environment(loader=FileSystemLoader(searchpath="."))

# Read template, render configs from arguments, and create namelist
template = env.get_template("namelist.template")

configs = {
    "aqm_date": aqm_date,
}

with open(f"namelist", "w") as f:
    f.write(template.render(**configs))
