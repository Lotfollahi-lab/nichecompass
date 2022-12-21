# Autotalker

## Copyright
This tool is developed in the Talavera-López Lab of the Institute of Computational Biology, Helmholtz Munich. The copyright holder for this project is the Talavera-López Lab. All rights reserved.

The tool will be open sourced once published.

## Poster
see /poster/autotalker_poster.pdf

## Setup
```conda env create -f env/environment.yml```

```python DeepLinc.py -e ./dataset/squidpy_seqfish_mouse_organogenesis/counts.csv -a ./dataset/squidpy_seqfish_mouse_organogenesis/adj.csv -c ./dataset/squidpy_seqfish_mouse_organogenesis/coords.csv -r ./dataset/squidpy_seqfish_mouse_organogenesis/cell_types.csv -n avg_n_neighbors_4 -i 40```
