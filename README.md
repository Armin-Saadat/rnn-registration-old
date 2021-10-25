# rnn-registration
Combining Unet with LSTMs to register 2D slices in a 3D image.

results doc:
https://docs.google.com/spreadsheets/d/1j2AjY6lBTuPASKP8m3J4J7YIdRMuD47HaOk7Smt5rrA/edit#gid=0


## Installation:
------------
Start by cloning this repositiory:
```
git clone https://github.com/Armin-Saadat/rnn-registration.git
cd rnn-registration
```
Create a virtual environment:
```
virtualenv myenv
source myenv/bin/activate
```
And install the dependencies:
```
pip install --upgrade pip  
pip install -r requirements.txt
pip install ./pystrum
pip install ./neurite
```

## Train:

```
python -m api.train -id=<name or number of the run> -device=<'cuda' or 'cpu> -epochs=<epochs>
```

## Evaluation:
  
```
python -m api.evaluate -id=<name or number of the train run> -snapshot=<name of the snapshot file> -device=<'cuda' or 'cpu>
```
