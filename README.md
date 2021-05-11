# sympy2quantized-sapt

group's repository on sympy based scripts to automate formulas derivation

## pre-configuration
at least python3.6 is required
```shell
python3 --version
```

install virtual enviroment module

```shell
python3 -m pip install virtualenv
```

in repository directory create venv

```shell
python3 -m virtualenv venv
```

activate virtual enviroment and install the dependencies

```shell
source venv/bin/activate
python3 -m pip install -r requirements.txt -r requirements_dev.txt
```

check if example ```SAPT``` sympy script runs
```shell
python3 examples/sapt_pol20.py
```

## install the development package
```shell
python3 -m pip install -e . --no-use-pep517
```

## run the unit tests
```shell
python3 -m pytest ./tests/
```

