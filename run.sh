#!/bin/bash

python a_01_4chan.download.json.py
python a_02_4chan.json.to.text.py 
python a_03_4chan.clean.text.py
python 00_clean.py
python 01_prepare_data.py
python 02_tune.py