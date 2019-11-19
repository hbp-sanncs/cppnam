CppNAM
======

CppNAM is a spiking neural network implementation of the Willshaw associative memory (BiNAM), implemented in C++ with Cypress/PyNN as a backend.

It serves as a potential benchmark for neuromorphic hardware systems and as a test bed for experiments with the Willshaw associative memory.

## Usage

First, build the project by executing in your target directory:

    git clone https://github.com/hbp-unibi/cppnam 
    mkdir cppnam/build && cd cppnam/build
    cmake ..
    make -j


To run the complete analysis pipeline, run the program in one of the following
forms:

    ./sp_binam <SIMULATOR> [<EXPERIMENT>]
    ./sp_binam <SIMULATOR>  <EXPERIMENT> NMPI

Where `SIMULATOR` is the simulator that should be used for execution (see below)
and `EXPERIMENT` is a JSON file describing the experiment that should be executed.
If NMPI is added at the end, the program will be executed through the HBP neuromorphic 
platform interface, which means it is executed on large scale hardware platforms.


## Simulators

Possible simulators are:

* `spikey`
* `nest`
* `nmmc1`
* `nmpm1`
* `ess`

## Experiment description documentation

The experiment description is read from a JSON-like file format. An example can be 
found in the folder experiments. The JSON is structured in subcategories:
in the 'data'-section the BiNAM related parameters are given. The 'network' part contains all
parameters for the neural network, where 'params' contains the neuron parameters, which
should be matching to the neuron-type given. Here we can also set the number of spikes 
representing a signal in input and output, as well as the time window between two successive 
pattern inputs. We can change the inter-spike-interval ('isi') in bursts and 'sigma_t' 
yields some Gaussian jitter in spike times, while 'sigma_offs' renders the offset of a spike train.
The third subcategory 'data_generator' contains parameters for the generation of the stored
patterns. The 'experiments' category contains the setting of parameters or sweeps. It is especially 
useful if you want to execute several simulations. The descriptor looks like this:

```javascript
	"experiments": {
		"Sweep_tau_m_syn" : {
			"params.tau_m" : {"min": 20, "max": 80, "count": 3},
			"network.weight" : {0.01,0.02},
			"params.tau_syn_E" : 1
		}
	}
```

It contains a name of the experiment, which will also be the file name in which the results 
will be stored. The parameter to change is given with 'subcategory.parameter', where only the 
last category is given (e.g. 'params'). Then we can pass a range, a list or single values.
The output file will change in dependence of what you pass: sweeps and list will give a csv, 
while only single values will yield the 'benchmark'-output with additional information 
like runtime in a nicely readable form.

## Authors

This project has been initiated as PyNAM by Andreas Stöckel in 2015 as part of his Masters Thesis
at Bielefeld University in the [Cognitronics and Sensor Systems Group](http://www.ks.cit-ec.uni-bielefeld.de/) which is
part of the [Human Brain Project, SP 9](https://www.humanbrainproject.eu/neuromorphic-computing-platform). 
This follow-up project has been realised by Christoph Jenzen 
and Andreas Stöckel, again within the [Cognitronics and Sensor Systems Group] at Bielefeld 
as part of our work for the [Human Brain Project, SP 9].

## License

This project and all its files are licensed under the
[GPL version 3](http://www.gnu.org/licenses/gpl.txt) unless explicitly stated
differently.




