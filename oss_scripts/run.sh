# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TASK=$1
AGENT=$2
ASSETS_ROOT=./ravens/environments/assets/

python ravens/demos.py  --assets_root=$ASSETS_ROOT --task=${TASK} --mode=train --n=1000
python ravens/demos.py  --assets_root=$ASSETS_ROOT --task=${TASK} --mode=test  --n=100

python ravens/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1
python ravens/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10
python ravens/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100
python ravens/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000

python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=1000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=2000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=5000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=10000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=20000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=40000

python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=1000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=2000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=5000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=10000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=20000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=40000

python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=1000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=2000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=5000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=10000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=20000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=40000

python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=1000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=2000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=5000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=10000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=20000
python ravens/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=40000

python ravens/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1
python ravens/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10
python ravens/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100
python ravens/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000
