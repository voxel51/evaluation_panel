# Evaluation Panel

A panel for evaluating your models in the FiftyOne App.

## Installation

Install latest:

```shell
fiftyone plugins download https://github.com/voxel51/evaluation_panel
```

Or, development install:

```shell
git clone https://github.com/voxel51/evaluation_panel

cd evaluation_panel
ln -s "$(pwd)" "$(fiftyone config plugins_dir)/evaluation_panel"
```

## Usage

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Then click the `+` next to the `Samples` tab and open the `Evaluation` panel.
